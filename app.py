# app.py
import os
import secrets
from datetime import datetime
from urllib.parse import urlencode
import pytz  # Importamos pytz para manejar zonas horarias

from flask import (
    Flask, render_template, request, redirect, url_for,
    session as flask_session, abort, flash
)
from flask import jsonify  # opcional
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import JSON  # cambia a db.JSON o db.Text si tu MySQL no soporta JSON nativo
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# ML/Reglas (opcional, solo para Burnout)
try:
    from ml.infer import MLModelPredictor
except Exception:
    MLModelPredictor = None

try:
    from ml.rules import classify_from_scores as burnout_rule_classifier
except Exception:
    burnout_rule_classifier = None

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------- Configuración MySQL ----------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:712002@127.0.0.1:3306/proyectocapstone?charset=utf8mb4"
)
app.config.update(
    SQLALCHEMY_DATABASE_URI=DATABASE_URL,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-pon-algo-largo"),
    SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True, "pool_recycle": 280},
    # SQLALCHEMY_ECHO=True,
)
db = SQLAlchemy(app)

# ---------- Invalidar sesión al reiniciar el servidor (solo añade este bloque) ----------
# Identificador único por arranque del servidor. Cambia en cada `flask run`.
app.config["BOOT_ID"] = secrets.token_hex(8)

@app.before_request
def _invalidate_session_on_restart():
    """
    Si el servidor se reinició, limpiar la cookie de sesión del cliente.
    Además, marcamos la sesión como NO permanente (se borra al cerrar el navegador).
    """
    flask_session.permanent = False
    if flask_session.get("_boot_id") != app.config["BOOT_ID"]:
        flask_session.clear()
        flask_session["_boot_id"] = app.config["BOOT_ID"]

# Definir la zona horaria de Lima (Perú)
LIMA_TZ = pytz.timezone('America/Lima')

# ===========================
#           MODELOS
# ===========================
class Participant(db.Model):
    __tablename__ = "participants"
    id = db.Column(db.Integer, primary_key=True)

    # Datos de cuenta
    email = db.Column(db.String(180), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # Perfil
    name = db.Column(db.String(180), nullable=False)
    age = db.Column(db.Integer)
    university = db.Column(db.String(180))
    career = db.Column(db.String(180))   # Carrera profesional
    cycle = db.Column(db.String(50))     # Ciclo/semestre (ej. "III", "5", "2025-1")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    sessions = db.relationship("TestSession", backref="participant", lazy=True)

    # Helpers de password
    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Test(db.Model):
    __tablename__ = "tests"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(32), unique=True, nullable=False)
    name = db.Column(db.String(180), nullable=False)
    version = db.Column(db.String(20), default="v1")
    likert_min = db.Column(db.Integer, default=1)
    likert_max = db.Column(db.Integer, default=5)
    description = db.Column(db.Text)

    sections = db.relationship("Section", backref="test", lazy=True, cascade="all, delete-orphan")
    questions = db.relationship("Question", backref="test", lazy=True, cascade="all, delete-orphan")


class Section(db.Model):
    __tablename__ = "sections"
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey("tests.id"), nullable=False, index=True)
    code = db.Column(db.String(64))
    title = db.Column(db.String(180), nullable=False)
    order = db.Column(db.Integer, default=0)

    questions = db.relationship("Question", backref="section", lazy=True)


class Question(db.Model):
    __tablename__ = "questions"
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey("tests.id"), nullable=False, index=True)
    section_id = db.Column(db.Integer, db.ForeignKey("sections.id"), nullable=True, index=True)
    number = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text, nullable=False)
    reversed = db.Column(db.Boolean, default=False)
    order = db.Column(db.Integer, default=0)

    __table_args__ = (
        db.UniqueConstraint("test_id", "number", name="uq_test_question_number"),
        db.Index("ix_questions_test_order", "test_id", "order"),
    )


class TestSession(db.Model):
    __tablename__ = "test_sessions"
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey("tests.id"), nullable=False, index=True)
    participant_id = db.Column(db.Integer, db.ForeignKey("participants.id"), nullable=False, index=True)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    ip_addr = db.Column(db.String(64))
    user_agent = db.Column(db.Text)
    version = db.Column(db.String(20), default="v1")
    score_total = db.Column(db.Integer)
    score_breakdown = db.Column(JSON)  # usa db.JSON si tu servidor no soporta JSON nativo
    e_tipo = db.Column("E_tipo", db.String(50))

    responses = db.relationship("Response", backref="session", cascade="all, delete-orphan", lazy=True)
    test = db.relationship("Test")


class Response(db.Model):
    __tablename__ = "responses"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("test_sessions.id"), nullable=False, index=True)
    question_id = db.Column(db.Integer, db.ForeignKey("questions.id"), nullable=False, index=True)
    value = db.Column(db.Integer, nullable=False)

    question = db.relationship("Question")

    __table_args__ = (
        db.UniqueConstraint("session_id", "question_id", name="uq_session_question"),
        # Accesos frecuentes
        db.Index("ix_responses_session", "session_id"),
        db.Index("ix_responses_question", "question_id"),
    )

# ===========================
#            SEED
# ===========================
def seed_tests_questions():
    if Test.query.count() > 0:
        return

    # --- TEST BURNOUT ---
    t_burn = Test(
        code="burnout",
        name="Test de Burnout Académico",
        likert_min=1, likert_max=5,
        description="Subescalas: personal, estudios, compañeros, profesores."
    )
    db.session.add(t_burn); db.session.flush()

    s_personal   = Section(test_id=t_burn.id, code="personal",   title="Burnout personal", order=1)
    s_estudios   = Section(test_id=t_burn.id, code="estudios",   title="Burnout en estudios", order=2)
    s_compas     = Section(test_id=t_burn.id, code="compañeros", title="Burnout con compañeros", order=3)
    s_profes     = Section(test_id=t_burn.id, code="profesores", title="Burnout con profesores", order=4)
    db.session.add_all([s_personal, s_estudios, s_compas, s_profes]); db.session.flush()

    burnout_items = [
        (s_personal, 1,  "¿Con qué frecuencia se siente cansado(a)?", False),
        (s_personal, 2,  "¿Con qué frecuencia se siente físicamente exhausto(a)?", False),
        (s_personal, 3,  "¿Con qué frecuencia se siente emocionalmente exhausto(a)?", False),
        (s_personal, 4,  "¿Con qué frecuencia piensa: “No aguanto más”?", False),
        (s_personal, 5,  "¿Con qué frecuencia se siente agotado(a)?", False),
        (s_personal, 6,  "¿Con qué frecuencia se siente débil y susceptible de enfermarse?", False),
        (s_estudios, 7,  "¿Se siente agotado(a) al final de un día de universidad?", False),
        (s_estudios, 8,  "¿Se siente exhausto(a) por la mañana al pensar en otro día en la universidad?", False),
        (s_estudios, 9,  "¿Siente que cada hora de clase/estudio es agotadora para usted?", False),
        (s_estudios, 10, "¿Tiene tiempo y energía para la familia y los amigos durante su tiempo libre?", True),
        (s_estudios, 11, "¿Sus estudios son emocionalmente agotadores?", False),
        (s_estudios, 12, "¿Se siente frustrado(a) con sus estudios?", False),
        (s_estudios, 13, "¿Se siente exhausto(a) de forma prolongada por sus estudios?", False),
        (s_compas, 14, "¿Le resulta difícil trabajar con sus compañeros(as) de estudio?", False),
        (s_compas, 15, "¿Siente que trabajar con compañeros(as) le drena la energía?", False),
        (s_compas, 16, "¿Le resulta frustrante trabajar con compañeros(as)?", False),
        (s_compas, 17, "¿Siente que da más de lo que recibe cuando trabaja con compañeros(as)?", False),
        (s_compas, 18, "¿Está cansado(a) de lidiar con los compañeros(as)?", False),
        (s_compas, 19, "¿A veces se pregunta cuánto tiempo más podrá seguir trabajando con los compañeros(as)?", False),
        (s_profes, 20, "¿Le resulta difícil tratar con los profesores?", False),
        (s_profes, 21, "¿Siente que tratar con los profesores le drena la energía?", False),
        (s_profes, 22, "¿Le resulta frustrante tratar con los profesores?", False),
        (s_profes, 23, "¿Siente que da más de lo que recibe cuando trata con los profesores?", False),
        (s_profes, 24, "¿Está cansado(a) de tratar con los profesores?", False),
        (s_profes, 25, "¿A veces se pregunta cuánto tiempo más podrá seguir tratando con los profesores?", False),
    ]
    for sec, num, txt, rev in burnout_items:
        db.session.add(Question(test_id=t_burn.id, section_id=sec.id, number=num, text=txt, reversed=rev, order=num))

    # --- TEST PSS-10 ---
    t_pss = Test(
        code="pss10",
        name="Escala de Estrés Percibido (PSS-10)",
        likert_min=1, likert_max=5,
        description="Ítems invertidos: 4,5,7,8"
    )
    db.session.add(t_pss); db.session.flush()
    s_pss = Section(test_id=t_pss.id, code="total", title="PSS-10", order=1)
    db.session.add(s_pss); db.session.flush()

    pss_items = [
        (1, "¿Con qué frecuencia te has molestado por algo que sucedió inesperadamente en el último mes?", False),
        (2, "¿Con qué frecuencia sentiste que no podías controlar las cosas importantes en tu vida en el último mes?", False),
        (3, "¿Con qué frecuencia te sentiste nervioso y \"estresado\" en el último mes?", False),
        (4, "¿Con qué frecuencia te sentiste confiado acerca de tu capacidad para manejar tus problemas personales en el último mes?", True),
        (5, "¿Con qué frecuencia sentiste que las cosas iban a tu favor en el último mes?", True),
        (6, "¿Con qué frecuencia descubriste que no podías hacer frente a todas las cosas que tenías que hacer en el último mes?", False),
        (7, "¿Con qué frecuencia pudiste controlar las irritaciones en tu vida en el último mes?", True),
        (8, "¿Con qué frecuencia sentiste que tenías todo bajo control en el último mes?", True),
        (9, "¿Con qué frecuencia te enfureciste por cosas que estaban fuera de tu control en el último mes?", False),
        (10,"¿Con qué frecuencia sentiste que las dificultades se acumulaban tanto que no podías superarlas en el último mes?", False),
    ]
    for num, txt, rev in pss_items:
        db.session.add(Question(test_id=t_pss.id, section_id=s_pss.id, number=num, text=txt, reversed=rev, order=num))

    # --- TEST SISCO SV-21 ---
    t_sisco = Test(
        code="sisco",
        name="Inventario SISCO de Estrés Académico (SV-21)",
        likert_min=1, likert_max=5,
        description="Tres dimensiones: Estresores, Reacciones, Estrategias de afrontamiento"
    )
    db.session.add(t_sisco); db.session.flush()

    s_estresores    = Section(test_id=t_sisco.id, code="estresores",    title="Estresores", order=1)
    s_reacciones    = Section(test_id=t_sisco.id, code="reacciones",    title="Reacciones físicas y psicológicas", order=2)
    s_afrontamiento = Section(test_id=t_sisco.id, code="afrontamiento", title="Estrategias de afrontamiento", order=3)
    db.session.add_all([s_estresores, s_reacciones, s_afrontamiento]); db.session.flush()

    sisco_items = [
        (s_estresores, 1,  "Sobrecarga de tareas y trabajos académicos.", False),
        (s_estresores, 2,  "La personalidad y carácter de los profesores.", False),
        (s_estresores, 3,  "Las evaluaciones de los profesores (exámenes, ensayos, trabajos de investigación, etc.).", False),
        (s_estresores, 4,  "El tipo de trabajo que te piden los profesores (consulta de mapas, fichas de trabajo, ensayos, mapas conceptuales, etc.).", False),
        (s_estresores, 5,  "No entender los temas que se abordan en la clase.", False),
        (s_estresores, 6,  "Participación en clase (responder a preguntas, exposiciones, etc.).", False),
        (s_estresores, 7,  "Tiempo limitado para hacer el trabajo.", False),
        (s_estresores, 8,  "Los compañeros de grupo progresan más rápido en tareas y/o trabajos académicos.", False),
        (s_reacciones, 9,  "Trastornos del sueño (insomnio o pesadillas).", False),
        (s_reacciones, 10, "Fatiga crónica (cansancio permanente).", False),
        (s_reacciones, 11, "Dolores de cabeza o migrañas.", False),
        (s_reacciones, 12, "Problemas de digestión, dolor abdominal o diarrea.", False),
        (s_reacciones, 13, "Rascarse, morderse las uñas, frotarse, etc.", False),
        (s_reacciones, 14, "Somnolencia o mayor necesidad de dormir.", False),
        (s_reacciones, 15, "Dolores musculares y/o contracturas.", False),
        (s_reacciones, 16, "Reacciones cutáneas (sarpullido, descamación, etc.).", False),
        (s_reacciones, 17, "Inquietud (incapacidad de relajarse y estar tranquilo).", False),
        (s_reacciones, 18, "Sentimientos de depresión y tristeza (decaído).", False),
        (s_reacciones, 19, "Ansiedad, angustia o desesperación.", False),
        (s_reacciones, 20, "Sentimiento de agresividad o aumento de irritabilidad.", False),
        (s_afrontamiento, 21, "Elaboración de un plan de ejecución de sus tareas.", False),
        (s_afrontamiento, 22, "Elogios a sí mismo.", False),
        (s_afrontamiento, 23, "Práctica religiosa (oraciones o asistencia a iglesia/templo).", False),
        (s_afrontamiento, 24, "Búsqueda de información sobre la situación.", False),
        (s_afrontamiento, 25, "Ventilación y confidencias (verbalización de la situación que preocupa).", False),
        (s_afrontamiento, 26, "Intenté sacar algo positivo o beneficioso de la situación estresante.", False),
        (s_afrontamiento, 27, "Consumo de sustancias (Café, energéticas, tabaco, etc.).", False),
        (s_afrontamiento, 28, "Practicar un pasatiempo (actividad física, leer, ver series, redes sociales, etc.).", False),
        (s_afrontamiento, 29, "Acompañarse de un ser querido (familia, mascotas, amigos, etc.).", False),
    ]
    for sec, num, txt, rev in sisco_items:
        db.session.add(Question(test_id=t_sisco.id, section_id=sec.id, number=num, text=txt, reversed=rev, order=num))

    db.session.commit()

# ===========================
#         INIT / CLI
# ===========================
with app.app_context():
    db.create_all()
    seed_tests_questions()

@app.cli.command("init-db")
def init_db_command():
    """Crea tablas y siembra datos base."""
    db.create_all()
    seed_tests_questions()
    print("DB inicializada y datos sembrados.")

# ===========================
#         UTILIDADES
# ===========================
def current_user():
    uid = flask_session.get("participant_id")
    return Participant.query.get(uid) if uid else None

def require_login(next_url: str | None = None):
    """Si no hay usuario en sesión, redirige a /login con ?next=..."""
    if not current_user():
        params = {"next": next_url or request.full_path}
        return redirect(url_for("login") + "?" + urlencode(params))
    return None

# Año actual y usuario para templates
@app.context_processor
def inject_globals():
    u = current_user()
    first = (u.name.split()[0] if u and u.name else None)
    return {
        "current_year": datetime.utcnow().year,
        "current_user": u,
        "current_user_first": first
    }

# ===========================
#           RUTAS
# ===========================
@app.route("/")
def inicio():
    tests = Test.query.order_by(Test.name).all()
    return render_template("inicio.html", tests=tests)

# ---- Registro (crear cuenta) ----
@app.route("/registro", methods=["GET", "POST"])
def registro():
    tests = Test.query.order_by(Test.name).all()
    if request.method == "GET":
        return render_template("registro.html", tests=tests)

    # POST: crear cuenta
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").lower().strip()
    password = request.form.get("password") or ""
    password2 = request.form.get("password2") or ""
    age = request.form.get("age")
    university = request.form.get("university")
    career = request.form.get("career")
    cycle = request.form.get("cycle")

    error = None
    if not name or not email or not password:
        error = "Nombre, correo y contraseña son obligatorios."
    elif password != password2:
        error = "Las contraseñas no coinciden."
    elif Participant.query.filter_by(email=email).first():
        error = "Ya existe una cuenta con ese correo."

    if error:
        flash(error, "error")
        return render_template("registro.html", tests=tests), 400

    p = Participant(
        name=name,
        email=email,
        age=int(age) if age else None,
        university=university,
        career=career,
        cycle=cycle,
    )
    p.set_password(password)
    db.session.add(p); db.session.commit()

    # Iniciar sesión
    flask_session["participant_id"] = p.id
    flash("Cuenta creada. ¡Bienvenido!", "ok")

    # Si vino con next, redirige allí
    next_url = request.args.get("next")
    return redirect(next_url or url_for("inicio"))

# ---- Login / Logout ----
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    email = (request.form.get("email") or "").lower().strip()
    password = request.form.get("password") or ""
    user = Participant.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        flash("Correo o contraseña incorrectos.", "error")
        return render_template("login.html"), 401

    flask_session["participant_id"] = user.id
    flash("Sesión iniciada.", "ok")
    next_url = request.args.get("next")
    return redirect(next_url or url_for("inicio"))

@app.route("/logout")
def logout():
    flask_session.pop("participant_id", None)
    flash("Sesión cerrada.", "ok")
    return redirect(url_for("inicio"))

# ===========================
#           PERFIL
# ===========================
@app.route("/perfil")
def perfil():
    need = require_login(next_url=url_for("perfil"))
    if need: return need
    return render_template("perfil.html")

@app.route("/perfil/datos")
def perfil_datos():
    need = require_login(next_url=url_for("perfil_datos"))
    if need: return need
    user = current_user()
    return render_template("perfil_datos.html", user=user)

@app.route("/perfil/tests")
def perfil_tests():
    need = require_login(next_url=url_for("perfil_tests"))
    if need: return need
    user = current_user()
    sessions = (
        TestSession.query
        .filter_by(participant_id=user.id)
        .order_by(TestSession.started_at.desc())
        .all()
    )
    return render_template("perfil_tests.html", sessions=sessions)

@app.route("/perfil/email", methods=["GET", "POST"])
def perfil_email():
    need = require_login(next_url=url_for("perfil_email"))
    if need: return need
    user = current_user()

    if request.method == "POST":
        new_email = (request.form.get("email") or "").lower().strip()
        confirm   = (request.form.get("email2") or "").lower().strip()
        password  = request.form.get("password") or ""

        if not new_email or not confirm or not password:
            flash("Completa todos los campos.", "error")
        elif new_email != confirm:
            flash("Los correos no coinciden.", "error")
        elif not user.check_password(password):
            flash("Contraseña incorrecta.", "error")
        elif Participant.query.filter(Participant.email == new_email, Participant.id != user.id).first():
            flash("Ese correo ya está en uso.", "error")
        else:
            user.email = new_email
            db.session.commit()
            flash("E-mail actualizado.", "ok")
            return redirect(url_for("perfil"))

    return render_template("perfil_email.html", user=user)

@app.route("/perfil/password", methods=["GET", "POST"])
def perfil_password():
    need = require_login(next_url=url_for("perfil_password"))
    if need: return need
    user = current_user()

    if request.method == "POST":
        current_pw = request.form.get("current_pw") or ""
        new_pw     = request.form.get("new_pw") or ""
        new_pw2    = request.form.get("new_pw2") or ""

        if not user.check_password(current_pw):
            flash("Tu contraseña actual no es correcta.", "error")
        elif len(new_pw) < 6:
            flash("La nueva contraseña debe tener al menos 6 caracteres.", "error")
        elif new_pw != new_pw2:
            flash("La confirmación no coincide.", "error")
        else:
            user.set_password(new_pw)
            db.session.commit()
            flash("Contraseña actualizada.", "ok")
            return redirect(url_for("perfil"))

    return render_template("perfil_password.html")

# ---- Tomar un test ----
@app.route("/test/<string:test_code>", methods=["GET", "POST"])
def test_run(test_code):
    test = Test.query.filter_by(code=test_code).first_or_404()

    # Exigir login en cualquier método
    need = require_login(next_url=url_for("test_run", test_code=test_code))
    if need: return need

    user = current_user()

    if request.method == "POST":
        # Asignar la hora de inicio solo si no se ha asignado previamente
        if not request.form.get("started_at"):
            started_at = datetime.now(LIMA_TZ)  # Obtener la hora actual en Lima
        else:
            started_at = request.form.get("started_at")  # Usar la hora pasada desde el formulario

        # Crear sesión del test para el usuario logueado
        sess = TestSession(
            test_id=test.id,
            participant_id=user.id,
            ip_addr=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
            version=test.version,
            started_at=started_at  # Asignar correctamente la hora de inicio
        )
        db.session.add(sess)
        db.session.flush()  # Guardar la sesión y obtener el ID para que podamos acceder al ID de la sesión

        # Inicializar el diccionario para el desglose de puntajes por sección
        score_breakdown = {}

        # Guardar respuestas y calcular el puntaje
        total_score = 0
        for q in test.questions:
            val = request.form.get(f"q{q.number}")
            if val is None:
                db.session.rollback()
                return f"Falta la respuesta {q.number}", 400
            try:
                response = Response(session_id=sess.id, question_id=q.id, value=int(val))
                db.session.add(response)

                # Sumar el valor de cada respuesta al puntaje total
                total_score += int(val)

                # Obtener la sección de la pregunta
                section_code = q.section.code

                # Si no existe una entrada para esta sección en score_breakdown, inicializamos su puntaje
                if section_code not in score_breakdown:
                    score_breakdown[section_code] = 0

                # Acumulamos el puntaje en el desglose por sección
                score_breakdown[section_code] += int(val)

            except ValueError:
                db.session.rollback()
                return f"Respuesta inválida para la pregunta {q.number}", 400

        # Guardar el puntaje total
        sess.score_total = total_score

        # Guardar el desglose de puntajes por sección en JSON
        sess.score_breakdown = score_breakdown

        # Solo cuando el test esté completado, asignamos la fecha de finalización
        completed_at = datetime.now(LIMA_TZ)  # Obtener la hora de finalización en Lima
        print(f"Completed at: {completed_at}")  # Imprimir la fecha de completado para depuración
        sess.completed_at = completed_at  # Asignar la hora de finalización solo cuando se complete

        db.session.commit()  # Guardar el puntaje total, el desglose y la fecha de finalización en la base de datos

        flask_session["session_id"] = sess.id  # Guardar la sesión activa
        return redirect(url_for("resultado", session_id=sess.id))

    return render_template("test.html", test=test)

@app.route("/resultado/<int:session_id>")
def resultado(session_id):
    sess = TestSession.query.get_or_404(session_id)
    return render_template(
        "resultado.html",
        session=sess,
        participant=sess.participant,
    )
    
# ===========================
#            MAIN
# ===========================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        seed_tests_questions()
    app.run(debug=True)
