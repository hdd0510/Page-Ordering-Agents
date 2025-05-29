import streamlit as st
import json
import os
import google.generativeai as genai
from typing import Optional, List, Dict, Any
import numpy as np
import tritonclient.grpc as grpcclient
import cv2
from PIL import Image
import io

# --- API Key Configuration ---
# Ensure GOOGLE_API_KEY is set as an environment variable
try:
    # Make sure to replace with your actual key or ensure the environment variable is set
    # For local development, you might hardcode it temporarily, but it's not recommended for production
    # Example: genai.configure(api_key="YOUR_ACTUAL_GOOGLE_API_KEY")
    if "GOOGLE_API_KEY" not in os.environ:
        # Attempt to load from .env file if it exists for local dev convenience
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass # dotenv not installed, that's fine, rely on direct env var

    if "GOOGLE_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        # If still not found, prompt user via Streamlit if this part of code is reached during app execution
        # However, for Streamlit, it's better to check this upfront or provide a way to input it.
        # For now, we assume it will be set before running, or the error below will trigger.
        st.error("🚨 **Error:** Environment variable GOOGLE_API_KEY is not set. The application might not work.")
        # st.stop() # This might be too aggressive if the app is just starting.
        # Let's try to configure and let it fail if key is truly missing.
        genai.configure(api_key="MISSING_KEY") # This will likely cause an error later if not set
except KeyError: # This specific block might not be hit if we try to configure directly
    st.error("🚨 **Error:** Environment variable GOOGLE_API_KEY is not set.")
    st.info("Please set your API key before running: `export GOOGLE_API_KEY='YOUR_API_KEY'` or add it to a `.env` file.")
    # st.stop() # Stop execution if key is critical and not found.
except Exception as e:
    st.error(f"🚨 Error configuring Generative AI API: {e}")
    # st.stop()


# --- Helper Functions (adapted from doctype_classify/split_page.py) ---

def sort_document_fragments(data_fragments: str) -> Optional[str]:
    """
    Sorts text fragments using the Gemini API.
    """
    prompt_template = f"""Sắp xếp các đoạn văn bản sau (`ID`, `Start Part`, `End Part`).

    **Ưu tiên cao nhất:** Dựa vào **số trang đơn lẻ** ở **cuối `End Part`** hoặc **đầu `Start Part`** để xác định thứ tự.

    Nếu không có số trang, hoặc cần xác nhận, hãy xem xét tính liền mạch nội dung giữa `End Part` và `Start Part` tiếp theo.

    Chỉ trả về danh sách các `ID` đã được sắp xếp theo định dạng sau, không thêm bất kỳ văn bản nào khác.

    **Dữ liệu:**
    {data_fragments}

    **Định dạng kết quả:**
    [ID_1, ID_2, ID_3, ...]"""

    try:
        # Ensure the model name is correct and you have access.
        # Consider making the model name configurable if needed.
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-preview-04-17',
            generation_config=genai.types.GenerationConfig(
                temperature=0.15,
                top_p=0.1
            )
        )
        response = model.generate_content(prompt_template)
        if response and response.text:
            return response.text.strip()
        else:
            st.warning("⚠️ Gemini API did not return text content.")
            if response:
                # Accessing prompt_feedback might differ based on the Gemini API version.
                # This is a common way, but verify with the library's documentation.
                feedback = getattr(response, 'prompt_feedback', 'No feedback available.')
                st.warning(f"   Reason for blocking (if any): {feedback}")
            return None
    except Exception as e:
        # Catching general exceptions is broad. Consider more specific ones if possible.
        st.error(f"🛑 Error calling Gemini API or processing response: {e}")
        # Log the full traceback for easier debugging if running in a controlled environment
        # import traceback
        # st.error(traceback.format_exc())
        return None

def extract_start_and_end(document, start_lines=5, end_lines=5):
    """Hàm nhận vào một chuỗi document, trả về đoạn đầu và đoạn cuối của document.
    start_lines: số dòng đầu muốn lấy
    end_lines: số dòng cuối muốn lấy
    """
    lines = document.split('\n')
    start_part = '\n'.join(lines[:start_lines])
    end_part = '\n'.join(lines[-end_lines:])
    return start_part, end_part

def process_page_data_for_sorting(ocr_results: List[Dict[str, Any]]) -> str:
    """Processes OCR results to create a context string for sorting."""
    context_all = ''
    for page_data in ocr_results:
        text_content = page_data.get("text", "")
        page_id = page_data.get("id") # This ID is the original sequence/filename ID

        start_part, end_part = extract_start_and_end(text_content)
        context = f'''
# ID: {page_id}
Start Part: \n{start_part}
End Part: \n{end_part}  
        '''
        context_all += context
    # print(context_all)
    return context_all

def call_triton_ocr(image_inp: np.ndarray, triton_url: str, model_name='OCR', input_name='input', output_names=['word_list', 'box_list', 'full_text'], data_type='UINT8'):
    """Calls Triton OCR service."""
    infer_input = grpcclient.InferInput(input_name, image_inp.shape, data_type)
    infer_input.set_data_from_numpy(image_inp)
    
    try:
        # Ensure Triton URL is correct and server is reachable.
        # Add a timeout if the client supports it, to prevent hanging indefinitely.
        client = grpcclient.InferenceServerClient(url=triton_url)
    except Exception as e:
        st.error(f"Failed to create Triton client for URL {triton_url}: {e}")
        raise # Re-raise to be caught by the caller
        
    outputs_req = [grpcclient.InferRequestedOutput(output_name) for output_name in output_names]
    
    try:
        response = client.infer(model_name=model_name, inputs=[infer_input], outputs=outputs_req)
        output_data = [response.as_numpy(output_name) for output_name in output_names]
        return output_data
    except Exception as e:
        st.error(f"Error during Triton inference for model {model_name}: {e}")
        # Consider more specific error handling or logging here.
        raise # Re-raise to be caught by run_ocr_pipeline_for_folder

def process_ocr_text(full_text_np_array: np.ndarray) -> str:
    """Decodes OCR text output."""
    if isinstance(full_text_np_array, np.ndarray) and full_text_np_array.size > 0:
        item = full_text_np_array.flat[0]
        if isinstance(item, bytes):
            try:
                return item.decode('utf-8')
            except UnicodeDecodeError:
                return f"Error decoding text (bytes: {item[:100]}...)"
        elif isinstance(item, np.bytes_): # Handle numpy.bytes_
            try:
                return item.decode('utf-8')
            except UnicodeDecodeError:
                return f"Error decoding text (np.bytes_: {item[:100]}...)"
        else:
            # If it's already a string or other type, convert to string
            return str(item)
    return "[No text data or not a recognized byte string]"

def run_ocr_pipeline_for_uploaded_files(uploaded_files: List[Any], triton_url: str) -> List[Dict[str, Any]]:
    """Runs OCR on uploaded image files, returns text and image data."""
    results = []
    if not uploaded_files:
        st.warning("No files were uploaded.")
        return results

    st.write(f"Found {len(uploaded_files)} image(s). Starting OCR process...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        current_id = i + 1  # User-friendly 1-based ID
        file_name = uploaded_file.name
        status_text.text(f"Processing {file_name} (ID: {current_id})...")

        try:
            # Read image data from uploaded file
            image_bytes = uploaded_file.getvalue()
            # Convert image bytes to OpenCV format (numpy array)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_inp = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image_inp is None:
                st.warning(f"Warning: Could not read image {file_name}. Skipping.")
                continue

            ocr_outputs = call_triton_ocr(image_inp, triton_url)
            
            processed_full_text = "[OCR output missing or empty]"
            if len(ocr_outputs) > 2 and ocr_outputs[2] is not None:
                full_text_np = ocr_outputs[2]
                processed_full_text = process_ocr_text(full_text_np)
            else:
                st.warning(f"Warning: Full text output not available or empty for {file_name}.")

            # Store image_bytes directly instead of path
            results.append({"id": current_id, "text": processed_full_text, "image_data": image_bytes, "name": file_name})
        except Exception as e:
            st.error(f"Error processing {file_name} with OCR: {e}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("OCR processing complete.")

    return results


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Document Image Sorter")
st.title("📄 Document Image Sorter & Visualizer")

# Instructions and Input in Sidebar
st.sidebar.header("⚙️ Configuration")
# image_folder_path = st.sidebar.text_input("Enter Full Path to Image Folder:", key="image_folder_path_input")
uploaded_files = st.sidebar.file_uploader(
    "Upload Document Images:",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'],
    accept_multiple_files=True,
    key="file_uploader_input"
)
triton_server_url = '10.124.68.252:8001'

# Gemini Model Selection (Optional)
# gemini_model_name = st.sidebar.selectbox("Choose Gemini Model:", 
#                                          ['gemini-1.5-flash-latest', 'gemini-pro', 'gemini-2.5-flash-preview-04-17'], 
#                                          index=0, key="gemini_model_select")


if st.sidebar.button("🚀 Sort and Display Images", key="sort_button"):
    # --- Input Validation ---
    if not uploaded_files: # Changed from image_folder_path
        st.sidebar.warning("⚠️ Please upload at least one image.")
    # elif not os.path.isdir(image_folder_path): # This check is no longer needed for direct uploads
    #     st.sidebar.error(f"❌ Invalid folder path: '{image_folder_path}'. Please ensure it's a full, correct path.")
    elif not triton_server_url:
        st.sidebar.warning("⚠️ Please enter the Triton server URL.")
    else:
        # --- Main Processing Logic ---
        st.header("📊 Processing Status & Results")
        with st.spinner("🔍 Step 1: Performing OCR on images... This might take a while for many images."):
            # ocr_results_with_paths = run_ocr_pipeline_for_folder(image_folder_path, triton_server_url)
            ocr_results_with_data = run_ocr_pipeline_for_uploaded_files(uploaded_files, triton_server_url)
        if not ocr_results_with_data: # Changed from ocr_results_with_paths
            st.warning("No images were successfully processed or found. Cannot proceed.")
            st.stop() # Stop execution if no OCR results

        # Create a mapping from original ID (1-based index) to image data for later display
        # image_map = {res["id"]: res["image_path"] for res in ocr_results_with_paths if "image_path" in res}
        image_map = {res["id"]: {"data": res["image_data"], "name": res["name"]} for res in ocr_results_with_data if "image_data" in res and "name" in res} # Store image data (bytes) and name
        if not image_map:
            st.error("Critical error: No image data was mapped from OCR results. Cannot display images.")
            st.stop()

        with st.spinner("📝 Step 2: Preparing text fragments for sorting..."):
            context_for_sorting = process_page_data_for_sorting(ocr_results_with_data) # Changed from ocr_results_with_paths
            # print(context_for_sorting)
        if not context_for_sorting.strip():
            st.error("No text context generated from OCR results. Cannot proceed with sorting.")
            st.stop()
        
        # Optionally display the context sent to Gemini for debugging
        # with st.expander("📋 View Context Sent for Sorting"):
        #     st.text_area("Context:", context_for_sorting, height=200, key="context_display_area")

        with st.spinner(f"🧠 Step 3: Calling Gemini API for sorting order..."):
            sorted_ids_api_str = sort_document_fragments(context_for_sorting) # Pass selected model if using selectbox

        if not sorted_ids_api_str:
            st.error("Failed to get a sorting order from the Gemini API. Please check API key and model access.")
            st.stop()
        
        # st.info(f"Raw Gemini Response (Sorted IDs): `{sorted_ids_api_str}`")

        # --- Parsing Gemini's Response ---
        predicted_ids = []
        try:
            # Standard JSON list parsing
            parsed_list = json.loads(sorted_ids_api_str)
            if isinstance(parsed_list, list):
                predicted_ids = [int(id_val) for id_val in parsed_list if isinstance(id_val, (int, str)) and str(id_val).strip().isdigit()]
            else:
                st.warning(f"⚠️ Gemini response was valid JSON but not a list: {sorted_ids_api_str}")
        except json.JSONDecodeError:
            # Fallback for non-JSON, comma-separated list like "1, 2, 3" or "[1, 2, 3]" (string)
            temp_str = sorted_ids_api_str.strip()
            if temp_str.startswith('[') and temp_str.endswith(']'):
                temp_str = temp_str[1:-1] # Remove brackets if present
            raw_ids = temp_str.split(',')
            predicted_ids = [int(id_val.strip()) for id_val in raw_ids if id_val.strip().isdigit()]
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred while parsing sorted IDs: {e}")
        
        if not predicted_ids:
            st.error("Could not extract any valid numeric IDs from Gemini's response. Cannot display sorted images.")
            st.stop()

        st.success(f"✅ Sorting order received. Predicted ID sequence: {predicted_ids}")

        # --- Displaying Sorted Images ---
        st.header("🖼️ Sorted Document Images")
        
        valid_predicted_images_info = []
        for pid in predicted_ids:
            if pid in image_map:
                # valid_predicted_images_info.append({"id": pid, "path": image_map[pid]})
                img_info_from_map = image_map[pid]
                valid_predicted_images_info.append({"id": pid, "data": img_info_from_map["data"], "name": img_info_from_map["name"]})
            else:
                st.warning(f"Predicted ID `{pid}` was not found among the processed images. It will be skipped.")
        
        if not valid_predicted_images_info:
            st.error("None of the predicted IDs correspond to processed images. Cannot display anything.")
            st.stop()

        # --- Display Configuration ---
        num_columns = st.slider("Number of columns for display:", 1, 5, 3, key="num_cols_slider")
        cols = st.columns(num_columns)
        
        for i, img_info in enumerate(valid_predicted_images_info):
            # image_path = img_info["path"]
            image_bytes_data = img_info["data"] # Changed from image_path to image_bytes_data
            original_id = img_info["id"]
            image_file_name = img_info["name"] # Get the original file name
            try:
                # if os.path.exists(image_path):
                image = Image.open(io.BytesIO(image_bytes_data)) # Use Pillow to open images from bytes
                with cols[i % num_columns]:
                    st.image(image, caption=f"Sorted: {i+1} (Orig. ID: {original_id} | File: {image_file_name})", use_container_width=True)
                # else:
                #      with cols[i % num_columns]:
                #         st.warning(f"Image not found at path: {image_path} (Orig. ID: {original_id})")
            except Exception as e:
                with cols[i % num_columns]:
                    st.error(f"Error loading image {image_file_name}: {e}")
        
        # st.balloons()
        st.success("🎉 Document sorting and display complete!")

# Sidebar Information
st.sidebar.markdown("---")
# st.sidebar.info(
#     "This app uses OCR to extract text from images in a folder, "
#     "sends text fragments to a Gemini AI model for sorting, "
#     "and then displays the images in the predicted order."
# # )
# st.sidebar.markdown(
#     "**Important Notes:**
# "
#     "- Ensure the `GOOGLE_API_KEY` environment variable is correctly set.
# "
#     "- The Triton OCR server must be running and accessible at the specified URL.
# "
#     # "- Provide the **full, absolute path** to the image folder."
#     "- Upload one or more images directly using the uploader."
# )

# To run: streamlit run app.py