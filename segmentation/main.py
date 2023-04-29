from flask import (Flask,render_template,request,redirect,
                   url_for)
from utils.auto_train_test_split import AutoTrainTestSplit
from utils.save_segment_data import SaveSegmentData
from auto_segmentation.auto_segmentation import AutoSegmentation
import json

app = Flask(__name__)

do_segment = SaveSegmentData()
# train,test,get_data = None,None,None


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
        global encoded_train
        global encoded_test
        train,test,encoded_train,encoded_test = train_test_split()
        print(train.shape,test.shape,encoded_train.shape,encoded_test.shape)
        
        
        return redirect(url_for('auto_segmentation'))
    
    
    
@app.route('/auto_segmentation')
def auto_segmentation():
    global get_data
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
    segmented_df = seg(cutoff=0.1)
    print(segmented_df)
     # add a new column with a radio input tag
    # val_dict = {'categories':segmented_df.categories.astype(str),
    #             'feature':segmented_df.feature.astype(str)}
    # json_value = json.dumps(val_dict)
    segmented_df['select'] = '<input type="radio" name="selected" value="' + 'feature:' + segmented_df.feature.astype(str) + ',' + 'categories:' + segmented_df.categories.astype(str) + '">'
 
     # convert the DataFrame to an HTML table string with escape=False
    table_html = segmented_df.to_html(escape=False)

    # pass the HTML string to the template
    return render_template('table.html', table_html=table_html)

    
@app.route('/process_form', methods=['POST'])
def process_form():
    # get the values of the checkboxes
    global train
    condition = request.form.get('selected')
    
    print(condition)
    segment = request.form.get('segment')
    if segment == 'done':
        segment_dict = do_segment.segmented_data_dict
        print(segment_dict)
        return "segment_dict"
    elif segment == 'segment_again':
        condition_dict = do_segment.get_feature_and_category_condition(condition)
        print(condition_dict)
        print('access',train.shape)
        print(train.index)
        do_segment.create_and_save_segment(train_df=train,condition_dict=condition_dict)
        print(train.shape,type(train))
        print(do_segment.segmented_data_dict.get('remaining_index'))

        train = train.loc[do_segment.segmented_data_dict.get('remaining_index')]
        print('remaining_df',train.shape)
        return redirect(url_for('auto_segmentation'))
    





if __name__ == '__main__':
    app.run(debug=True)