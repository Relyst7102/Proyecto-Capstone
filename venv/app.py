# app.py
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session as flask_session
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

# === ML / reglas ===
from ml.infer import MLModelPredictor
from ml.rules import classify_from_scores

# Carga variables de entorno (.env)
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')

# ---------- Config DB ----------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:712002@127.0.0.1:3306/test_burnout?charset=utf8mb4"
)
app.config.update(
    SQLALCHEMY_DATABASE_URI=DATABASE_URL,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-pon-algo-largo"),
    SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True, "pool_recycle": 280},
)

db = SQLAlchemy(app)

# ---------- Modelos ----------
class Participant(db.Model):
    __tablename__ = "participants"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(180), nullable=False)
    age = db.Column(db.Integer)
    university = db.Column(db.String(180))
    email = db.Column(db.String(180))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sessions = db.relationship("TestSession", backref="participant", lazy=True)

class TestSession(db.Model):
    __tablename__ = "test_sessions"
    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(db.Integer, db.ForeignKey("participants.id"), nullable=False, index=True)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    ip_addr = db.Column(db.String(64))
    user_agent = db.Column(db.Text)
    version = db.Column(db.String(20), default="v1")

    score_total = db.Column(db.Integer)
    score_personal = db.Column(db.Integer)
    score_studies = db.Column(db.Integer)
    score_peers = db.Column(db.Integer)
    score_teachers = db.Column(db.Integer)

    responses = db.relationship("Response", backref="session", cascade="all, delete-orphan", lazy=True)

class Question(db.Model):
    __tablename__ = "questions"
    id = db.Column(db.Integer, primary_key=True)  # 1..25
    text = db.Column(db.Text, nullable=False)
    domain = db.Column(db.String(32))             # personal, estudios, compañeros, profesores
    reversed = db.Column(db.Boolean, default=False)

class Response(db.Model):
    __tablename__ = "responses"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("test_sessions.id"), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey("questions.id"), nullable=False)
    value = db.Column(db.Integer, nullable=False)

    question = db.relationship("Question")

    __table_args__ = (
        db.UniqueConstraint("session_id", "question_id", name="uq_session_question"),
        db.Index("ix_responses_session", "session_id"),
        db.Index("ix_responses_question", "question_id"),
    )

# ---------- Semilla de preguntas ----------
QUESTIONS_SEED = [
    (1,  "¿Con qué frecuencia se siente cansado(a)?", "personal", False),
    (2,  "¿Con qué frecuencia se siente físicamente exhausto(a)?", "personal", False),
    (3,  "¿Con qué frecuencia se siente emocionalmente exhausto(a)?", "personal", False),
    (4,  "¿Con qué frecuencia piensa: “No aguanto más”?", "personal", False),
    (5,  "¿Con qué frecuencia se siente agotado(a)?", "personal", False),
    (6,  "¿Con qué frecuencia se siente débil y susceptible de enfermarse?", "personal", False),
    (7,  "¿Se siente agotado(a) al final de un día de universidad?", "estudios", False),
    (8,  "¿Se siente exhausto(a) por la mañana al pensar en otro día en la universidad?", "estudios", False),
    (9,  "¿Siente que cada hora de clase/estudio es agotadora para usted?", "estudios", False),
    (10, "¿Tiene tiempo y energía para la familia y los amigos durante su tiempo libre?", "estudios", True),
    (11, "¿Sus estudios son emocionalmente agotadores?", "estudios", False),
    (12, "¿Se siente frustrado(a) con sus estudios?", "estudios", False),
    (13, "¿Se siente exhausto(a) de forma prolongada por sus estudios?", "estudios", False),
    (14, "¿Le resulta difícil trabajar con sus compañeros(as) de estudio?", "compañeros", False),
    (15, "¿Siente que trabajar con compañeros(as) le drena la energía?", "compañeros", False),
    (16, "¿Le resulta frustrante trabajar con compañeros(as)?", "compañeros", False),
    (17, "¿Siente que da más de lo que recibe cuando trabaja con compañeros(as)?", "compañeros", False),
    (18, "¿Está cansado(a) de lidiar con los compañeros(as)?", "compañeros", False),
    (19, "¿A veces se pregunta cuánto tiempo más podrá seguir trabajando con los compañeros(as)?", "compañeros", False),
    (20, "¿Le resulta difícil tratar con los profesores?", "profesores", False),
    (21, "¿Siente que tratar con los profesores le drena la energía?", "profesores", False),
    (22, "¿Le resulta frustrante tratar con los profesores?", "profesores", False),
    (23, "¿Siente que da más de lo que recibe cuando trata con los profesores?", "profesores", False),
    (24, "¿Está cansado(a) de tratar con los profesores?", "profesores", False),
    (25, "¿A veces se pregunta cuánto tiempo más podrá seguir tratando con los profesores?", "profesores", False),
]

def seed_questions():
    if Question.query.count() == 0:
        for qid, text, dom, rev in QUESTIONS_SEED:
            db.session.add(Question(id=qid, text=text, domain=dom, reversed=rev))
        db.session.commit()

# ---------- Inicialización para Flask 3 ----------
# Ejecuta la creación de tablas y el seed al cargar la app.
# Con el recargador en debug, el módulo se importa 2 veces; este guard evita doble seed.
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    with app.app_context():
        db.create_all()
        seed_questions()

# ---------- Predictor ML (carga el último modelo de /models) ----------
PREDICTOR = MLModelPredictor(models_dir="models")

# ---------- Utilidades ----------
def current_session():
    sid = flask_session.get("session_id")
    return TestSession.query.get(sid) if sid else None

def compute_scores(values_by_qid):
    doms = {
        "personal":  list(range(1,7)),
        "estudios":  list(range(7,14)),
        "compañeros": list(range(14,20)),
        "profesores": list(range(20,26)),
    }
    total = 0
    subt = {k: 0 for k in doms}
    for q in Question.query.order_by(Question.id).all():
        v = int(values_by_qid.get(q.id, 0))
        if q.reversed:
            v = 6 - v  # invierte 1..5
        total += v
        subt[q.domain] += v
    return {
        "total": total,
        "personal": subt["personal"],
        "studies": subt["estudios"],
        "peers": subt["compañeros"],
        "teachers": subt["profesores"],
    }

# ---------- Rutas ----------
@app.route("/")
def inicio():
    return render_template("inicio.html")

@app.route("/registro", methods=["GET"])
def registro():
    return render_template("registro.html")

@app.route("/test", methods=["GET", "POST"])
def test_burnout():
    if request.method == "POST":
        # Si viene q1, es envío del test; si no, viene del registro
        if request.form.get("q1"):
            sess = current_session()
            if not sess:
                p = Participant(name="Anónimo")
                db.session.add(p); db.session.commit()
                sess = TestSession(
                    participant_id=p.id,
                    ip_addr=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
                db.session.add(sess); db.session.commit()
                flask_session["session_id"] = sess.id

            # borra respuestas previas (por si reenvían)
            Response.query.filter_by(session_id=sess.id).delete()

            values = {}
            for qid in range(1, 26):
                val = request.form.get(f"q{qid}")
                if val is None:
                    return "Faltan respuestas.", 400
                values[qid] = int(val)
                db.session.add(Response(session_id=sess.id, question_id=qid, value=int(val)))

            scores = compute_scores(values)
            sess.score_total = scores["total"]
            sess.score_personal = scores["personal"]
            sess.score_studies = scores["studies"]
            sess.score_peers = scores["peers"]
            sess.score_teachers = scores["teachers"]
            sess.completed_at = datetime.utcnow()
            db.session.commit()

            return redirect(url_for("resultado", session_id=sess.id))

        # POST desde registro
        name = request.form.get("nombre") or "Sin nombre"
        age = request.form.get("edad")
        university = request.form.get("universidad")
        email = request.form.get("correo")

        p = Participant(
            name=name,
            age=int(age) if age else None,
            university=university,
            email=email,
        )
        db.session.add(p); db.session.commit()

        sess = TestSession(
            participant_id=p.id,
            ip_addr=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
        )
        db.session.add(sess); db.session.commit()

        flask_session["session_id"] = sess.id
        return render_template("test.html")

    # GET normal
    return render_template("test.html")

@app.route("/resultado/<int:session_id>")
def resultado(session_id):
    sess = TestSession.query.get_or_404(session_id)

    # Respuestas de la sesión (para ML)
    responses = Response.query.filter_by(session_id=session_id).order_by(Response.question_id).all()
    answers = {f"q{r.question_id}": r.value for r in responses}

    # Predicción por ML (si hay modelo)
    ml_pred = PREDICTOR.predict(answers)  # dict con label/proba/version/model

    # Clasificación por reglas (siempre disponible)
    rule_label, perc = classify_from_scores(
        sess.score_personal, sess.score_studies, sess.score_peers,
        sess.score_teachers, sess.score_total
    )

    return render_template(
        "resultado.html",
        session=sess,
        participant=sess.participant,
        responses=responses,
        ml_pred=ml_pred,
        rule_label=rule_label,
        perc=perc,
    )

# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True)
