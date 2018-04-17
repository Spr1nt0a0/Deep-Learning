#-*-coding:utf-8-*-
import requests
import json
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

url = 'http://music.163.com/weapi/v1/resource/comments/R_SO_4_385965?csrf_token=e1fe15efef4247d07236a5ce33c6a104'

headers = {
   'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
   'Referer':'http://music.163.com/song?id=385965',
   'Origin':'http://music.163.com',
   'Host':'music.163.com'
}
#加密数据，直接拿过来用
user_data = {
   'params': '18JgVc7TTw9t/iPo5/L26WsAVOjFBzXDuSTFkrlwIHooija/5x9V3u+JOAtILZn5XU4DwwxodEOewuGAiqrVIpoxSXk09wrzRzf+lW7UOolMHDFo7kGpVbqTWlgI0FSnmii8EemXCtP9zA6zlWG6VXEGeNvOlTxgP5Y8M64NpRK9k7wS5IVrsIbtXG+HDofB92Mr2DlIcalg7xOB7g9hxPoGQyiaM1BQBeb/I3FXIsQ=',
   'encSecKey': 'b126938e89fde22245acacd398a355df300e53e5c2b227169a150f29510deefcca807f403cc2d962b4afa1d07c5261ca1c8123ef965e8d9b91b7bb01c7eaf2caf96de683d2ec6a08a5f4091e51338ec8631f8a4663c893d5c11bf11c53a596fc41b68e62fa17e38cfdc1a276d439773e88df302db689d56a762d106540a0d25b'
}

response = requests.post(url,headers=headers,data=user_data)

data = json.loads(response.text)
hotcomments = []
for hotcommment in data['hotComments']:
   item = {
       'nickname':hotcommment['user']['nickname'],
       'content':hotcommment['content'],
       'likedCount':hotcommment['likedCount']
   }
   hotcomments.append(item)

#获取评论用户名，内容，以及对应的获赞数
content_list = [content['content'] for content in hotcomments]
nickname = [content['nickname'] for content in hotcomments]
liked_count = [content['likedCount'] for content in hotcomments]

from pyecharts import Bar

bar = Bar("热评中点赞数示例图")
bar.add( "点赞数",nickname, liked_count, is_stack=True,mark_line=["min", "max"],mark_point=["average"])
bar.render()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

d = path.dirname(__file__)
alice_coloring = np.array(Image.open(path.join(d, "111.jpg")))

stopwords = set(STOPWORDS)
stopwords.add("said")

content_text = " ".join(content_list)

wc = WordCloud(font_path=r"C:\simhei.ttf",background_color="white", max_words=2000, mask=alice_coloring,
               stopwords=stopwords, max_font_size=40, random_state=42)
wc.generate(content_text)

image_colors = ImageColorGenerator(alice_coloring)

#wordcloud = WordCloud(font_path=r"C:\simhei.ttf",max_words=200).generate(content_text)

plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.figure()

plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.imshow(alice_coloring, cmap=plt.cm.gray, interpolation="bilinear")
plt.axis("off")
plt.show()