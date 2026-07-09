from flask import Flask, render_template, request, send_file
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

app = Flask(__name__)

# Prediction History
history = []

# Last Prediction (for PDF)
last_news = ""
last_prediction = ""
last_confidence = ""

# Load trained model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global last_news, last_prediction, last_confidence

    news = request.form["news"]

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)[0]

    confidence = model.predict_proba(news_vector)[0]
    confidence_score = round(max(confidence) * 100, 2)

    if prediction == 1:
        result = "✅ REAL NEWS"
    else:
        result = "❌ FAKE NEWS"

    # Save last prediction for PDF
    last_news = news
    last_prediction = result
    last_confidence = confidence_score

    # Save History
    history.insert(0, {
        "news": news[:80] + ("..." if len(news) > 80 else ""),
        "prediction": result,
        "confidence": confidence_score
    })

    # Keep only last 5 predictions
    history[:] = history[:5]

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence_score,
        news=news,
        history=history
    )


@app.route("/download")
def download():

    doc = SimpleDocTemplate("Prediction_Report.pdf")
    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>Fake News Detection Report</b>", styles["Title"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph(f"<b>Prediction:</b> {last_prediction}", styles["Normal"]))
    story.append(Paragraph(f"<b>Confidence:</b> {last_confidence}%", styles["Normal"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>News:</b>", styles["Heading2"]))
    story.append(Paragraph(last_news, styles["BodyText"]))

    story.append(Paragraph("<br/>", styles["Normal"]))
    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        )
    )

    doc.build(story)

    return send_file("Prediction_Report.pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)