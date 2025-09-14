import os
import requests
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NvidiaAIClient:
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables")
        
        self.header_auth = f"Bearer {self.api_key}"
        self.invoke_url = "https://ai.api.nvidia.com/v1/cv/hive/ai-generated-image-detection"
        self.assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
    
    def upload_asset(self, file_path, description="Input Image"):
        """Upload large files as assets to NVIDIA API"""
        # Create asset
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.header_auth,
            "accept": "application/json",
        }
        
        # Determine content type based on file extension
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            content_type = "image/jpeg"
        elif file_path.lower().endswith('.png'):
            content_type = "image/png"
        else:
            content_type = "image/jpeg"  # default
        
        payload = {
            "contentType": content_type,
            "description": description
        }
        
        response = requests.post(self.assets_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Asset creation failed: {response.status_code} - {response.text}")
        
        asset_info = response.json()
        upload_url = asset_info["uploadUrl"]
        asset_id = asset_info["assetId"]
        
        # Upload file data
        with open(file_path, "rb") as f:
            file_headers = {
                "Content-Type": content_type,
                "x-amz-meta-nvcf-asset-description": description,
            }
            upload_response = requests.put(upload_url, data=f, headers=file_headers)
            if upload_response.status_code != 200:
                raise Exception(f"Asset upload failed: {upload_response.status_code} - {upload_response.text}")
        
        return asset_id
    
    def detect_ai_generated_image(self, file_path):
        """Detect if an image is AI-generated using NVIDIA API"""
        # Read image content
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Convert to base64
        image_b64 = base64.b64encode(content).decode()
        
        # Determine content type based on file extension
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            content_type = "image/jpeg"
        elif file_path.lower().endswith('.png'):
            content_type = "image/png"
        else:
            content_type = "image/jpeg"  # default
        
        # Determine upload method based on size
        if len(image_b64) < 180_000:  # Small file - send directly
            # Send directly in payload
            payload = {
                "input": [f"data:{content_type};base64,{image_b64}"]
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": self.header_auth,
                "Accept": "application/json",
            }
        else:  # Large file - upload as asset
            asset_id = self.upload_asset(file_path, "Input Image")
            
            payload = {
                "input": [f"data:{content_type};asset_id,{asset_id}"]
            }
            headers = {
                "Content-Type": "application/json",
                "NVCF-INPUT-ASSET-REFERENCES": asset_id,
                "Authorization": self.header_auth,
            }
        
        # Call NVIDIA API
        response = requests.post(self.invoke_url, headers=headers, json=payload)
        
        # Return result
        if response.status_code == 200:
            result = response.json()
            # Process the result to extract the AI probability
            try:
                ai_prob = 0.0
                if "data" in result and len(result["data"]) > 0:
                    data_item = result["data"][0]
                    # Correct field name based on actual API response
                    if "is_ai_generated" in data_item:
                        ai_prob = data_item["is_ai_generated"]
                    elif "aiGenerated" in data_item:
                        ai_prob = data_item["aiGenerated"]
                    elif "ai_generated" in data_item:
                        ai_prob = data_item["ai_generated"]
                
                # Return both the raw result and the interpreted probability
                processed_result = {
                    "success": True, 
                    "data": result,
                    "ai_probability": ai_prob,
                    "is_ai_generated": ai_prob > 0.5
                }
                return processed_result
            except Exception as parse_error:
                # If we can't parse the specific format, return the raw result
                error_result = {
                    "success": True, 
                    "data": result,
                    "ai_probability": 0.0,
                    "is_ai_generated": False,
                    "parse_error": str(parse_error)
                }
                return error_result
        else:
            error_text = response.text
            raise Exception(f"API error: {response.status_code} - {error_text}")