import os

from flask import Flask, request, jsonify, render_template
from plankton_predict import predict_img, implement_roi_image, detect_and_save_contours

app = Flask(__name__)

@app.route('/')
def index():
    return render_template(template_name_or_list='opening.html')

@app.route('/action', methods=['GET'])
def action():
    return render_template(template_name_or_list='action.html')

@app.route('/opening')
def delete_upload():
    for file in os.listdir('static/uploads'):
        if file != 'original_image.jpg' and file != 'predicted_mask.jpg' and file != 'output_image.jpg':
            os.remove(os.path.join('static/uploads', file))
            
    return render_template(template_name_or_list='opening.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({
            "error": "No file part"
        }), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            "error": "No selected file"
        }), 400
        
    if file:
        img_path = os.path.join('static/uploads', file.filename)
        file.save(img_path)
        return jsonify({
            "img_path": img_path
        }), 200
    else:
        return jsonify({
            "error": "File extension not allowed"
        }), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_path = data['img_path']
    model_option = data['model_option']
    llm_option = data['llm_option']
    
    roi, _ = implement_roi_image(img_path)
    actual_class, probability_class, response = predict_img(model_option, llm_option, roi)
    
    return jsonify({
        "img_path": img_path,
        "actual_class": actual_class,
        "probability_class": probability_class,
        "response": response
    }), 200

@app.route('/result')
def result():
    img_path = request.args.get('img_path')
    actual_class = request.args.getlist('actual_class')[0].split(",")
    probability_class = request.args.getlist('probability_class')[0].split(",")
    response = request.args.get('response')
    
    detect_and_save_contours(
        "static/uploads/original_image.jpg", 
        "static/uploads/predicted_mask.jpg", 
        "static/uploads/output_image.jpg"
    )

    output_path = os.path.join('static/uploads', "output_image.jpg")
    
    if img_path:
        return render_template(
            template_name_or_list='result.html', 
            img_path=output_path,
            class1=actual_class[0], 
            probability1=f'{float(probability_class[0]):.6f}',
            class2=actual_class[1], 
            probability2=f'{float(probability_class[1]):.6f}',
            class3=actual_class[2], 
            probability3=f'{float(probability_class[2]):.6f}',
            response=response
        )
    else:
        return render_template(template_name_or_list='opening.html')

if __name__ == "__main__":
    app.run(debug=True)