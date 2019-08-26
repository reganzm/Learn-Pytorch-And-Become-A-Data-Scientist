import flask
from gevent import pywsgi
import json
from fastai_bert_classification import predict_learner
from flask import Flask,request,render_template
#实例化出server对象
server=flask.Flask(__name__) 
def response_wrapper(keep_dict = False,**kargs):
    """
     将字典类型包装成json
     params: keep_dict: Whether or not keep dict type
    """
    return kargs if(keep_dict) else json.dumps(kargs,ensure_ascii=False)

if __name__ == "__main__":
    learner = predict_learner()
    @server.route('/fastai_bert_classification/predict',methods=['post','get'])
    def preview_file_():
        try:
            if request.method == 'GET':
                return render_template('input.html')
            else:            
                message = flask.request.values.get('message')   
                result = learner.predict(message)
                return response_wrapper(category_label=result[1].item(),neg_probability=result[2][0].item(),pos_probability=result[2][1].item())
        except Exception as e:
            print(e)
            return e.args[0].__str__()
    wsgi_server = pywsgi.WSGIServer(("localhost", 1314), server)
    wsgi_server.serve_forever() 