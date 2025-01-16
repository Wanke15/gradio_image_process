import base64

import gradio as gr

import cv2

from zhipuai import ZhipuAI

client = ZhipuAI(api_key="xxx.xxx")

def llm_image_tag(image):
    image_base64 = ""
    response = client.chat.completions.create(
        model="glm-4v-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                        你是一位资深图像视觉设计师，你需要分析图片中的视觉元素，如具体的物品名称、人物、景观、帅哥、美女、老奶奶等，并给图片打上相应标签（尽量用中文，但不必强求），用于图像搜索场景，给出简洁的说明及原因，输出格式为json，示例如下：
                        [{"label": "遮阳伞", "description": "木制躺椅，配有蓝色坐垫", "score": 0.8]
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    }
                ]
            }
        ]
    )
    content = response.choices[0].message.content
    res = content.replace("```json", "").replace("```", "")
    # res = json.loads(res)
    # res = json.dumps(res, indent=4, ensure_ascii=False)
    return res


def process_image(image):
   _, buffer = cv2.imencode('.jpg', image)
   base64_str = base64.b64encode(buffer).decode('utf-8')

   res = llm_image_tag(base64_str)
   return res

# Gradio Interface
iface = gr.Interface(
   title="Build an Image Understanding Engine",
   description="by Jeff",
   fn=process_image,
   inputs=gr.Image(label="Input Image"),
   outputs=gr.Json(label="Result"),
)
iface.launch()
