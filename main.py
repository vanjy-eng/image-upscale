import sys
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

def enhance_image(image_path):
    print("Loading model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path="/Users/vangelis/Documents/Code Snippets/image-upscale/weights/RealESRGAN_x4plus.pth", #'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.4.0/RealESRGAN_x4plus.pth',
        model=model,
        device='cpu'
    )

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    try:
        output, _ = upsampler.enhance(img, outscale=4)
        enhanced_image_path = image_path.replace('.jpg', '_enhanced.jpg')
        cv2.imwrite(enhanced_image_path, output)
        return enhanced_image_path
    except RuntimeError as error:
        print('Error:', error)
        return None

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python enhance_image.py <image_path>")
    #     sys.exit(1)

    # image_path = sys.argv[1]
    print("Enhancing image...")
    enhanced_image_path = enhance_image("/Users/vangelis/Documents/Code Snippets/image-upscale/images/ronaldo.jpg")
    if enhanced_image_path:
        print(enhanced_image_path)
