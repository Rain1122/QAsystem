from flask import Flask, render_template, request
from DocumentRetrievalModel import DocumentRetrievalModel as DRM
from ProcessedQuestion import ProcessedQuestion as PQ
from QuestionRetrievalModel import QuestionRetrievalModel as QRM

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get1")
def get_bot_response():
    userText = request.args.get('msg')

    #
    try:
        datasetFile = open("./dataset/database.txt", "r")
    except FileNotFoundError:
        # print("Bot> Oops! I am unable to locate \"" + datasetName + "\"")
        # add email return function
        exit()

    paragraphs = []
    for para in datasetFile.readlines():
        if (len(para.strip()) > 0):
            paragraphs.append(para.strip())

    drm = DRM(paragraphs, True, True)

    # greetPattern = re.compile("^\ *((hi+)|((good\ )?morning|evening|afternoon)|(he((llo)|y+)))\ *$", re.IGNORECASE)

    userQuery = userText

    if (not len(userQuery) > 0):
        print("You need to ask something")
    # elif greetPattern.findall(userQuery):
    #     response = "Hello!"
    else:
        pq = PQ(userQuery, True, False, True)
        response = drm.query(pq)
        res = response
        return res

@app.route("/get2")
def question():
    userText = request.args.get('msg')

    #
    try:
        questionFile = open("./dataset/question.txt", "r")
    except FileNotFoundError:
        # print("Bot> Oops! I am unable to locate \"" + datasetName + "\"")
        # add email return function
        exit()

    paragraphs = []

    for para in questionFile.readlines():
        if (len(para.strip()) > 0):
            paragraphs.append(para.strip())

    qrm = QRM(paragraphs, True, True)

    # greetPattern = re.compile("^\ *((hi+)|((good\ )?morning|evening|afternoon)|(he((llo)|y+)))\ *$", re.IGNORECASE)

    userQuery = userText
    pq = PQ(userQuery, True, False, True)
    response = qrm.query(pq)
    ques = response
    return ques


if __name__ == "__main__":
    app.run()
