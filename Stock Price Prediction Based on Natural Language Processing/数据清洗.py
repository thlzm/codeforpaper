import json
import jieba
def keyWordCleaning(dt):
    words_cut = jieba.cut(dt['keywords'] )
    dt['keywords'] = ','.join(words_cut)
    print(dt['keywords'] )
    return dt


with open(r'D:\desktop\research\数据集\new2016zh\news2016zh_valid.json','r',encoding='utf8')as fp:
    for f in fp:

        json_data = json.loads(f)

        # 对keyword进行清洗
        # print(json_data)
        json_data = keyWordCleaning(json_data)

        # 将清洗后数据追加进文件
        with open(r'D:\desktop\research\数据集\new2016zh\news2016zh_valid_clean.json', 'a', encoding='utf8')as file:
            file.write(json.dumps(json_data))
        # print('这是文件中的json数据：',json_data)
        # print('-'*30)
        # print('这是读取到文件数据的数据类型：', type(json_data))

        break
