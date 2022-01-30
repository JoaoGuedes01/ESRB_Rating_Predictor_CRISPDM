from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder='HTML')
model = joblib.load('model/model.pkl')
port = int(os.environ.get('PORT', 33507))

print('Model Imported')
print('Atributes:\n\nClasses:{}\n\nTraining Samples:{}\n\nTraining Features:{}\n\nIterations:{}\n\nLayers:{}\n\nOutputs:{}'.format(model.classes_,model.t_,model.n_features_in_, model.n_iter_, model.n_layers_, model.n_outputs_))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

## Rotas Backend
@app.route('/sendData', methods=['POST'])
def sendData():
    json_data = request.json
    no_descriptors = json_data['no_descriptors']
    strong_language = json_data['strong_language']
    blood_and_gore = json_data['blood_and_gore']
    fantasy_violence = json_data['fantasy_violence']
    blood = json_data['blood']
    mild_fantasy_violence = json_data['mild_fantasy_violence']
    strong_sexual_content = json_data['strong_sexual_content']
    sexual_themes = json_data['sexual_themes']
    intense_violence = json_data['intense_violence']
    suggestive_themes = json_data['suggestive_themes']
    violence = json_data['violence']
    simulated_gambling = json_data['simulated_gambling']
    sexual_content = json_data['sexual_content']
    language = json_data['language']
    mild_blood = json_data['mild_blood']
    mild_suggestive_themes = json_data['mild_suggestive_themes']
    crude_humor = json_data['crude_humor']
    mild_violence = json_data['mild_violence']
    mild_lyrics = json_data['mild_lyrics']
    cartoon_violence = json_data['cartoon_violence']
    alcohol_reference = json_data['alcohol_reference']
    lyrics = json_data['lyrics']
    drug_reference = json_data['drug_reference']
    use_of_alcohol = json_data['use_of_alcohol']
    partial_nudity = json_data['partial_nudity']
    nudity = json_data['no_descriptors']
    mild_cartoon_violence = json_data['no_descriptors']
    animated_blood = json_data['no_descriptors']

    newData = np.array([[no_descriptors,strong_language,blood_and_gore,fantasy_violence,blood,mild_fantasy_violence,strong_sexual_content,sexual_themes,
    intense_violence,suggestive_themes,violence,simulated_gambling,sexual_content,language,mild_blood,mild_suggestive_themes,crude_humor,mild_violence,
    mild_lyrics,cartoon_violence,alcohol_reference,lyrics,drug_reference,use_of_alcohol,partial_nudity,nudity,mild_cartoon_violence,animated_blood]])

    predictedRating = model.predict(newData)[0];
    print(predictedRating)
    finalres = ''
    desc = ''
    if predictedRating == 0:
       finalres = 'E'
       desc = 'O Rating E é isto e aquilo'
    if predictedRating == 1:
       finalres = 'ET'
       desc = 'O Rating ET é isto e aquilo'
    if predictedRating == 2:
       finalres = 'T'
       desc = 'O Rating T é isto e aquilo'
    if predictedRating == 3:
       finalres = 'M'
       desc = 'O Rating M é isto e aquilo'

    return {'status':200, 'message':finalres, 'desc':desc}

if __name__ == '__main__':
    app.run(port);


