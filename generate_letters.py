from PIL import Image, ImageDraw, ImageFont
import os

def generate_letter_images(letters, font_path, image_size=(28, 28), font_size=24, output_dir='letters'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for letter in letters:
        image = Image.new('L', image_size, color=0)  # 创建黑色背景的灰度图像
        draw = ImageDraw.Draw(image)
        
        # 加载字体
        font = ImageFont.truetype(font_path, font_size)
        
        # 获取字体大小
        text_width, text_height = draw.textsize(letter, font=font)
        
        # 计算文本位置，使其居中
        position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
        
        # 绘制字母
        draw.text(position, letter, fill=255, font=font)
        
        # 保存图像
        image.save(os.path.join(output_dir, f'{letter}.png'))
        print(f'Saved {letter}.png')

# 生成字母图像
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'  # 请根据实际字体文件路径修改
generate_letter_images(letters, font_path)
