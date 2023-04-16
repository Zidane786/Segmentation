from flask import (Flask,render_template,request,redirect,
                   url_for)
from utils.auto_train_test_split import AutoTrainTestSplit 
from auto_segmentation.auto_segmentation import AutoSegmentation


app = Flask(__name__)



models_available = ['Decision Tree','Random Forest']
file_type_available = ["excel","csv"]



@app.route('/')
def home():
    
    return render_template('home.html',models_available=models_available,file_type_available=file_type_available)

@app.route('/redirect',methods = ['POST'])
def submit_form():
    if request.method == 'POST':
        global get_data
        get_data = dict(request.form)
        train_test_split = AutoTrainTestSplit(get_data)
        global test
        global train
        train,test = train_test_split()
        print(train.shape,test.shape)
        
        
        return redirect(url_for('auto_segmentation'))
    
    
    
@app.route('/auto_segmentation')
def auto_segmentation():
    print(get_data)
    # train = request.args.get('train')
    print(train)
    print(type(train))
    seg = AutoSegmentation(df = train,
                           target=get_data.get('target'),
                           event=get_data.get('event'),
                           no_of_category = get_data.get('no_of_category'),
                           no_of_bucket=get_data.get('no_of_bucket'))
    seg.start_segmentation()
    segmented_df = seg()
    print(segmented_df)
     # add a new column with a radio input tag
    segmented_df['select'] = '<input type="radio" name="selected" value="{index}">'
    
     # convert the DataFrame to an HTML table string with escape=False
    table_html = segmented_df.to_html(escape=False)

    # pass the HTML string to the template
    return render_template('table.html', table_html=table_html)

    
@app.route('/process_form', methods=['POST'])
def process_form():
    # get the values of the checkboxes
    selected = request.form.getlist('selected')
    return selected
    





if __name__ == '__main__':
    app.run(debug=True)