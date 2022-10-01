import pyrebase
import datetime

firebaseConfig = {
"apiKey": "AIzaSyDQ6ezR_fLVHBRdMLPnF9Imnwtsw_iW8fc",
"authDomain": "summarizerproj-doc-store.firebaseapp.com",
"projectId": "summarizerproj-doc-store",
"storageBucket": "summarizerproj-doc-store.appspot.com",
"messagingSenderId": "979283029162",
"appId": "1:979283029162:web:fd9dd3efa93a9bb16df4d3",
"measurementId": "G-BBJLGXEVN2",
"databaseURL" : "https://console.firebase.google.com/u/2/project/summarizerproj-doc-store/database/summarizerproj-doc-store-default-rtdb/data/~2F"
}

users = []
def storage(user, doc_type, document, time_app_ran):
    path_of_uploading_file = "user_" + doc_type + ".txt"
    path_to_waca_filestore = user.replace(' ','') + '__' + doc_type + str(time_app_ran) +".txt"

    with open(path_of_uploading_file, "w", encoding='utf-8') as f:
        f.write(document)        


    firebase = pyrebase.initialize_app(firebaseConfig)
    storage = firebase.storage()

    #upload a file
    storage.child(path_to_waca_filestore).put(path_of_uploading_file)

    #download a file
    # storage.child(path_to_waca_filestore).download(f"gs://waca-filestore.appspot.com/{path_to_waca_filestore}", 'test.txt')
