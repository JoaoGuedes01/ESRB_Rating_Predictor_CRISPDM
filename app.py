from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np


app = Flask(__name__, template_folder='HTML')
# model = joblib.load('model/model.pkl')

# print('Model Imported')
# print('Atributes:\n\nClasses:{}\n\nTraining Samples:{}\n\nTraining Features:{}\n\nIterations:{}\n\nLayers:{}\n\nOutputs:{}'.format(model.classes_,model.t_,model.n_features_in_, model.n_iter_, model.n_layers_, model.n_outputs_))

# Rotas Frontend
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def test():
    return render_template('test.html')

@app.route('/prevCustDash')
def dashPrevCust():
    return render_template('dash1.html')

@app.route('/prevsDash')
def dashPrevs():
    return render_template('dash2.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

# Rotas Backend
@app.route('/sendData', methods=['POST'])
def sendData():
   # Get Data from Request
    json_data = request.json

    # Select Model
    json_model = json_data['model']
    print(' selecting models/{}.pkl'.format(json_model))
    model = joblib.load('models/{}.pkl'.format(json_model))

    # ML Parameters from Request
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

   # New ML Data For Prediction
    newData = np.array([[no_descriptors, strong_language, blood_and_gore, fantasy_violence, blood, mild_fantasy_violence, strong_sexual_content, sexual_themes,
                         intense_violence, suggestive_themes, violence, simulated_gambling, sexual_content, language, mild_blood, mild_suggestive_themes, crude_humor, mild_violence,
                         mild_lyrics, cartoon_violence, alcohol_reference, lyrics, drug_reference, use_of_alcohol, partial_nudity, nudity, mild_cartoon_violence, animated_blood]])

    predictedRating = model.predict(newData)[0]
    finalres = ''
    desc = ''

    # Send Response with the prediction result and description
    if predictedRating == 0:
        finalres = 'E'
        desc = 'Os títulos classificados como E (Todos) têm conteúdo que pode ser adequado para pessoas com 6 anos ou mais. Os títulos nesta categoria podem conter um mínimo de desenho animado, fantasia ou violência leve e/ou uso infrequente de linguagem leve.'
    if predictedRating == 1:
        finalres = 'ET'
        desc = 'A classificação ESRB E10+ (Todos com mais de 10 anos) indica que o conteúdo é geralmente adequado para maiores de 10 anos e pode conter mais desenhos animados, fantasia ou violência leve ou temas sugestivos mínimos. Nenhuma dessas classificações, nem qualquer outra classificação etária atribuída pela ESRB, no entanto, indica que um jogo ou aplicativo é direcionado a Crianças para fins da COPPA.'
    if predictedRating == 2:
        finalres = 'T'
        desc = 'Os títulos classificados como T (Teen) têm conteúdo que pode ser adequado para maiores de 13 anos. Os títulos nesta categoria podem conter violência, temas sugestivos, humor grosseiro, sangue mínimo, jogos de azar simulados e/ou uso infrequente de linguagem forte.'
    if predictedRating == 3:
        finalres = 'M'
        desc = 'Os títulos classificados como M (Adulto) têm conteúdo que pode ser adequado para pessoas com 17 anos ou mais. Os títulos nesta categoria podem conter violência intensa, sangue e violência, conteúdo sexual e/ou linguagem forte.'

    return {'status': 200, 'message': finalres, 'desc': desc}

# Start App in port 33507
if __name__ == '__main__':
    app.run(port=33507)
