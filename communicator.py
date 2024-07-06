from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin


import json, os
import networkx as nx
from networkx.readwrite import json_graph
from generate import layout_generate

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

cuda_idx = -1


@app.route('/getGraphData', methods=['GET'])
@cross_origin()
def getUploadGraphData():
    file_name = request.args.get("name")
    data = parse_upload_data(file_name)
    print("parse_data", data)
    path = "public/upload_" + file_name
    os.remove(path)  # 删除本地的文件
    return data


@app.route('/uploadGraphData', methods=['POST'])
@cross_origin()
def file_upload():
    requ_data = request.files.get('file')
    file_name = requ_data.filename
    file_path = "public/upload_" + file_name
    requ_data.save(file_path)

    return "upload ok"


@app.route('/getDiffsuionData', methods=['GET'])
@cross_origin()
def data_diffsuion():
    global cuda_idx
    cuda_idx = (cuda_idx +1) % 7
    data = json.loads(request.args.get("dataParam"))['data']
    print("data", data)
    return_data = layout_generate(data, cuda_idx)

    return return_data


def parse_upload_data(filename):
    filepath = "public/upload_" + filename

    # # 读取GraphML文件
    graph = nx.read_graphml(filepath)  # 替换为实际的文件路径

    # 将GraphML转换为JSON格式
    data = json_graph.node_link_data(graph)

    json_data = json.dumps(data)

    return json_data


if __name__ == "__main__":
    print('run 0.0.0.0:15000')
    app.run(host='0.0.0.0', port=15000)
