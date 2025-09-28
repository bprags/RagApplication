"""
Example usage of the Persistent Task Agent for website color extraction tasks.

This example demonstrates how to use the enhanced ValidationEngine
to validate color extraction from website screenshots or images.
"""

import os
from typing import List, Tuple
from agent_framework import (
    PersistentTaskAgent, AgentConfig, LLMInterface,
    ColorExtractor, ColorParser, ValidationEngine
)

# Example LLM interface for OpenAI (you'll need to implement your actual LLM)
class ExampleLLMInterface:
    """Example LLM interface - replace with your actual LLM implementation."""
    
    def __init__(self, api_key: str = "your-api-key"):
        self.api_key = api_key
        # Initialize your LLM client here
    
    def send_request(self, prompt: str, context: str = None) -> str:
        """Send request to your LLM and return response."""
        # Replace this with actual LLM API call
        # This is just a mock response for demonstration
        if "extract" in prompt.lower() and "color" in prompt.lower():
            return """
            Based on the website image, I can identify the following dominant colors:
            
            1. Primary Blue: RGB(0, 123, 255) - Used for headers and primary buttons
            2. White: RGB(255, 255, 255) - Background color
            3. Dark Gray: RGB(52, 58, 64) - Primary text color
            4. Light Gray: RGB(248, 249, 250) - Secondary background areas
            5. Success Green: RGB(40, 167, 69) - Success indicators and buttons
            6. Warning Orange: RGB(255, 193, 7) - Warning elements
            7. Danger Red: RGB(220, 53, 69) - Error states and danger buttons
            
            These colors form a cohesive design system typical of modern web applications.
            """
        else:
            return "I'll help you with that task."
    
    def is_available(self) -> bool:
        return True


def extract_website_colors(image_path: str, llm_interface: LLMInterface) -> dict:
    """
    Extract colors from a website screenshot using the persistent agent.
    
    Args:
        image_path: Path to the website screenshot
        llm_interface: LLM interface for processing
        
    Returns:
        Dictionary with extraction results
    """
    
    # Configure agent for color extraction tasks
    config = AgentConfig(
        max_attempts=8,  # More attempts for complex color extraction
        retry_delay=1.0,
        log_level="INFO",
        enable_learning=True
    )
    
    # Create the persistent agent
    agent = PersistentTaskAgent(llm_interface, config)
    
    # Define the color extraction task
    task_description = f"""
    Analyze the website screenshot and extract the dominant colors used in the design.
    
    Please provide:
    1. The main brand/theme colors with RGB values
    2. Background colors
    3. Text colors
    4. Accent colors used for buttons, links, etc.
    5. Any other significant colors in the design
    
    Format each color as RGB(r, g, b) and include a brief description of where it's used.
    """
    
    # Execute the task with image input
    result = agent.execute_task(
        task_description,
        input_data={"image_path": image_path}
    )
    
    return result


def batch_color_extraction(image_paths: List[str], llm_interface: LLMInterface):
    """
    Extract colors from multiple website images in batch.
    
    Args:
        image_paths: List of paths to website screenshots
        llm_interface: LLM interface for processing
    """
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING IMAGE {i}/{len(image_paths)}")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        try:
            result = extract_website_colors(image_path, llm_interface)
            results.append({
                'image_path': image_path,
                'result': result
            })
            
            print(f"✅ Success: {result['success']}")
            print(f"Attempts: {result['attempts_made']}")
            print(f"Execution Time: {result['execution_time']:.2f}s")
            
            if result['success']:
                # Parse and display extracted colors
                parser = ColorParser()
                colors = parser.parse_colors_from_text(result['final_response'])
                
                print(f"🎨 Extracted Colors ({len(colors)} found):")
                for j, color in enumerate(colors, 1):
                    color_name = parser.color_to_name(color)
                    print(f"  {j}. RGB{color} ({color_name})")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r.get('result', {}).get('success', False))
    print(f"Images processed: {len(results)}")
    print(f"Successful extractions: {successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    return results


def analyze_color_accuracy(image_path: str, extracted_colors: List[Tuple[int, int, int]]):
    """
    Analyze the accuracy of extracted colors against the actual image.
    
    Args:
        image_path: Path to the image file
        extracted_colors: List of RGB color tuples from LLM
    """
    
    print(f"\n🔍 COLOR ACCURACY ANALYSIS")
    print(f"Image: {os.path.basename(image_path)}")
    print("-" * 40)
    
    try:
        # Extract actual colors from image
        extractor = ColorExtractor()
        actual_colors = extractor.extract_dominant_colors(image_path, num_colors=10)
        
        print(f"Actual dominant colors ({len(actual_colors)} found):")
        parser = ColorParser()
        for i, color in enumerate(actual_colors, 1):
            color_name = parser.color_to_name(color)
            print(f"  {i}. RGB{color} ({color_name})")
        
        print(f"\nExtracted colors ({len(extracted_colors)} found):")
        for i, color in enumerate(extracted_colors, 1):
            color_name = parser.color_to_name(color)
            print(f"  {i}. RGB{color} ({color_name})")
        
        # Analyze matches
        print(f"\n🎯 MATCHING ANALYSIS:")
        matches = []
        threshold = 30  # Distance threshold for good match
        
        for i, extracted in enumerate(extracted_colors, 1):
            closest_actual, distance = extractor.find_closest_color(extracted, actual_colors)
            is_good_match = distance < threshold
            
            status = "✅ Good match" if is_good_match else "⚠️ Poor match"
            print(f"  {i}. RGB{extracted} -> RGB{closest_actual} (distance: {distance:.1f}) {status}")
            
            matches.append(is_good_match)
        
        accuracy = sum(matches) / len(matches) * 100 if matches else 0
        print(f"\n📊 Overall Accuracy: {accuracy:.1f}% ({sum(matches)}/{len(matches)} good matches)")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def main():
    """Main example demonstrating website color extraction."""
    
    print("🎨 WEBSITE COLOR EXTRACTION EXAMPLE")
    print("="*60)
    
    # Initialize LLM interface (replace with your actual implementation)
    llm = ExampleLLMInterface()
    
    # Example 1: Single image color extraction
    print("\n📸 SINGLE IMAGE EXTRACTION")
    
    # Note: Replace with actual image paths
    sample_image = "/path/to/website_screenshot.png"
    
    if os.path.exists(sample_image):
        result = extract_website_colors(sample_image, llm)
        
        if result['success']:
            # Parse extracted colors
            parser = ColorParser()
            extracted_colors = parser.parse_colors_from_text(result['final_response'])
            
            # Analyze accuracy
            analyze_color_accuracy(sample_image, extracted_colors)
    else:
        print(f"Sample image not found: {sample_image}")
        print("Create a sample image or update the path to test the extraction.")
    
    # Example 2: Batch processing
    print("\n\n📚 BATCH PROCESSING EXAMPLE")
    
    sample_images = [
        "/path/to/website1.png",
        "/path/to/website2.png", 
        "/path/to/website3.png"
    ]
    
    # Filter to existing images
    existing_images = [img for img in sample_images if os.path.exists(img)]
    
    if existing_images:
        batch_results = batch_color_extraction(existing_images, llm)
    else:
        print("No sample images found. Please provide actual website screenshot paths.")
    
    # Example 3: Color parsing demonstration
    print("\n\n🧩 COLOR PARSING DEMONSTRATION")
    
    sample_llm_responses = [
        """
        The website uses these main colors:
        - Primary: RGB(0, 123, 255) for headers
        - Background: RGB(255, 255, 255) 
        - Text: RGB(33, 37, 41)
        - Accent: #28a745 for success elements
        """,
        """
        Color palette analysis:
        1. Blue theme: (0, 100, 200)
        2. White backgrounds: (255, 255, 255)  
        3. Dark text: (50, 50, 50)
        4. Red accents: #dc3545
        """,
        """
        Main colors found: blue rgb(70, 130, 180), white #ffffff, 
        gray (128, 128, 128), and green accent RGB(40, 167, 69).
        """
    ]
    
    parser = ColorParser()
    
    for i, response in enumerate(sample_llm_responses, 1):
        print(f"\nSample Response {i}:")
        print(f"Text: {response.strip()}")
        
        colors = parser.parse_colors_from_text(response)
        print(f"Parsed Colors: {len(colors)} found")
        
        for j, color in enumerate(colors, 1):
            color_name = parser.color_to_name(color)
            print(f"  {j}. RGB{color} -> {color_name}")


if __name__ == "__main__":
    main()
