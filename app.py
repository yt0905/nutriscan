import os, io, json, base64, gdown
from datetime import datetime, timedelta, date, timezone

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from PIL import Image

app = Flask(__name__)
app.config.update(
    SECRET_KEY               = os.environ.get('SECRET_KEY', 'nutriscan-fyp2-secret'),
    SQLALCHEMY_DATABASE_URI  = os.environ.get('DATABASE_URL', 'sqlite:///nutriscan.db').replace('postgresql://', 'postgresql+psycopg2://'),
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    MAX_CONTENT_LENGTH       = 16 * 1024 * 1024,
)

db = SQLAlchemy(app)
lm = LoginManager(app)
lm.login_view    = 'login'
lm.login_message = 'Please log in to continue.'

NUTRITION  = pd.read_csv('nutrition_final.csv').set_index('food_label')
with open('class_names.json') as f:
    CLASS_NAMES = json.load(f)

# local file paths for the models
MODEL_PATHS = {
    'MobileNetV2':    'model/mobilenetv2_final.keras',
    'EfficientNetB0': 'model/efficientnetb0_final.keras',
    'ResNet50':       'model/resnet50_final.keras',
}

# google drive file IDs — download only when model is needed
MODEL_DRIVE_IDS = {
    'model/mobilenetv2_final.keras':    '1iegYKWPX2LX4DWIifSbOT3NaqFTwZoNZ',
    'model/efficientnetb0_final.keras': '1BuqUsd2E1LxErqcDDXNBMUHL-jd7pTgs',
    'model/resnet50_final.keras':       '1cqeNur7A4Ns-cgaNv1QpL2FQL59myNBB',
}

ALL_MODES = ['MobileNetV2', 'EfficientNetB0', 'ResNet50', 'Ensemble']
MODE_ACC  = {
    'MobileNetV2':    '64.3%',
    'EfficientNetB0': '70.8%',
    'ResNet50':       '70.5%',
    'Ensemble':       '74.4% star',
}

def malaysia_now():
    return datetime.now(timezone.utc) + timedelta(hours=8)

def malaysia_today():
    return malaysia_now().date()

_model_cache = {}

def load_single_model(name):
    if name not in _model_cache:
        import tensorflow as tf
        import gc
        if len(_model_cache) > 0:
            _model_cache.clear()
            gc.collect()
        path = MODEL_PATHS[name]
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f'Downloading {path}...')
            gdown.download(id=MODEL_DRIVE_IDS[path], output=path, quiet=False)
        print(f'  Loading {name}...')
        _model_cache[name] = tf.keras.models.load_model(path)
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _model_cache[name].predict(dummy, verbose=0)
        print(f'  {name} ready.')
    return _model_cache[name]

def preload_all_models():
    print('Skipping preload — models will load on first request.')

def _preprocess_arr(arr, model_name):
    import tensorflow as tf
    a = arr.copy()
    if model_name == 'MobileNetV2':
        a = tf.keras.applications.mobilenet_v2.preprocess_input(a)
    elif model_name == 'EfficientNetB0':
        a = tf.keras.applications.efficientnet.preprocess_input(a)
    elif model_name == 'ResNet50':
        a = tf.keras.applications.resnet50.preprocess_input(a)
    return np.expand_dims(a, 0)

def predict_single(img_bytes, model_name):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    arr = np.array(img.resize((224, 224)), dtype=np.float32)
    return load_single_model(model_name).predict(_preprocess_arr(arr, model_name), verbose=0)[0]

def run_inference(img_bytes, mode):
    import time

    if mode in MODEL_PATHS:
        t0 = time.time()
        preds = predict_single(img_bytes, mode)
        timings = {mode: round((time.time() - t0) * 1000, 1)}
        return preds, timings

    elif mode == 'Ensemble':
        timing_results = {}
        pred_arrays = {}
        for name in MODEL_PATHS:
            t0 = time.time()
            p = predict_single(img_bytes, name)
            timing_results[name] = round((time.time() - t0) * 1000, 1)
            pred_arrays[name] = p
        stacked = np.stack(list(pred_arrays.values()), axis=0)
        avg_preds = np.mean(stacked, axis=0)
        return avg_preds, timing_results

    raise ValueError(f'Unknown mode: {mode}')


class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(64),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    meals         = db.relationship('MealLog', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, pw):   self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)

class MealLog(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    food_label = db.Column(db.String(100), nullable=False)
    food_name  = db.Column(db.String(150), nullable=False)
    calories   = db.Column(db.Float, nullable=False)
    protein    = db.Column(db.Float, nullable=False)
    carbs      = db.Column(db.Float, nullable=False)
    fat        = db.Column(db.Float, nullable=False)
    fiber      = db.Column(db.Float, nullable=False)
    serving_g  = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.String(50))
    meal_type  = db.Column(db.String(20), default='Breakfast')
    logged_at  = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc) + timedelta(hours=8))

@lm.user_loader
def load_user(uid): return User.query.get(int(uid))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('scan'))
    if request.method == 'POST':
        u = User.query.filter_by(email=request.form['email'].strip().lower()).first()
        if u and u.check_password(request.form['password']):
            login_user(u, remember=True)
            return redirect(request.args.get('next') or url_for('scan'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('scan'))
    if request.method == 'POST':
        username = request.form['username'].strip()
        email    = request.form['email'].strip().lower()
        password = request.form['password']
        if not username or not email or not password:
            flash('All fields are required.', 'error')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
        elif User.query.filter_by(username=username).first():
            flash('Username already taken.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        else:
            u = User(username=username, email=email)
            u.set_password(password)
            db.session.add(u); db.session.commit()
            login_user(u, remember=True)
            flash(f'Welcome, {username}!', 'success')
            return redirect(url_for('scan'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user(); return redirect(url_for('login'))

@app.route('/')
@login_required
def index(): return redirect(url_for('scan'))

@app.route('/scan')
@login_required
def scan():
    food_names = sorted(NUTRITION['food_name_display'].tolist())
    return render_template('scan.html', models=ALL_MODES, mode_acc=MODE_ACC, food_names=food_names)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files or not request.files['image'].filename:
        return jsonify({'error': 'No image uploaded.'}), 400
    img_bytes = request.files['image'].read()
    mode      = request.form.get('model', 'Ensemble')
    if mode not in ALL_MODES:
        return jsonify({'error': f'Unknown mode: {mode}'}), 400
    try:
        preds, timings = run_inference(img_bytes, mode)
        top5_idx = np.argsort(preds)[::-1][:5]
        top5 = [{
            'label':      CLASS_NAMES[i],
            'name':       NUTRITION.loc[CLASS_NAMES[i], 'food_name_display'] if CLASS_NAMES[i] in NUTRITION.index else CLASS_NAMES[i].replace('_', ' ').title(),
            'confidence': round(float(preds[i]) * 100, 1),
        } for i in top5_idx]
        best_label = top5[0]['label']
        nutrition  = None
        if best_label in NUTRITION.index:
            row = NUTRITION.loc[best_label]; s = float(row['serving_size_g']) / 100
            nutrition = {
                'food_label': best_label,
                'food_name':  str(row['food_name_display']),
                'serving_g':  float(row['serving_size_g']),
                'calories':   round(float(row['calories_per_100g']) * s, 1),
                'protein':    round(float(row['protein_per_100g'])  * s, 1),
                'carbs':      round(float(row['carbs_per_100g'])    * s, 1),
                'fat':        round(float(row['fat_per_100g'])      * s, 1),
                'fiber':      round(float(row['fiber_per_100g'])    * s, 1),
            }
        img_b64 = base64.b64encode(img_bytes).decode()
        total_ms = round(sum(timings.values()), 1)
        return jsonify({
            'top5':      top5,
            'nutrition': nutrition,
            'model_used': mode,
            'image_src': f'data:image/jpeg;base64,{img_b64}',
            'timings':   timings,
            'total_ms':  total_ms,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/log_meal', methods=['POST'])
@login_required
def log_meal():
    d    = request.get_json()
    meal = MealLog(
        user_id=current_user.id, food_label=d['food_label'], food_name=d['food_name'],
        calories=d['calories'], protein=d['protein'], carbs=d['carbs'],
        fat=d['fat'], fiber=d['fiber'], serving_g=d['serving_g'],
        model_used=d.get('model_used', ''), meal_type=d.get('meal_type', 'Breakfast'),
    )
    db.session.add(meal); db.session.commit()
    return jsonify({'success': True, 'id': meal.id})

@app.route('/delete_meal/<int:meal_id>', methods=['POST'])
@login_required
def delete_meal(meal_id):
    meal = MealLog.query.get_or_404(meal_id)
    if meal.user_id != current_user.id: return jsonify({'error': 'Forbidden'}), 403
    db.session.delete(meal); db.session.commit()
    return jsonify({'success': True})

@app.route('/edit_meal/<int:meal_id>', methods=['POST'])
@login_required
def edit_meal(meal_id):
    meal = MealLog.query.get_or_404(meal_id)
    if meal.user_id != current_user.id:
        return jsonify({'error': 'Forbidden'}), 403
    d = request.get_json()

    # Use new food label if provided, otherwise keep existing
    label  = d.get('food_label', meal.food_label)
    new_s  = float(d.get('serving_g', meal.serving_g))

    if new_s < 10:
        return jsonify({'error': 'Serving size too small (min 10g)'}), 400

    if label not in NUTRITION.index:
        return jsonify({'error': 'Food not found'}), 404

    # Always recalculate nutrition from scratch using label + serving
    row = NUTRITION.loc[label]
    s   = new_s / 100
    meal.food_label = label
    meal.food_name  = str(row['food_name_display'])
    meal.calories   = round(float(row['calories_per_100g']) * s, 1)
    meal.protein    = round(float(row['protein_per_100g'])  * s, 1)
    meal.carbs      = round(float(row['carbs_per_100g'])    * s, 1)
    meal.fat        = round(float(row['fat_per_100g'])      * s, 1)
    meal.fiber      = round(float(row['fiber_per_100g'])    * s, 1)
    meal.serving_g  = new_s

    if 'meal_type' in d and d['meal_type']:
        meal.meal_type = d['meal_type']

    if 'logged_date' in d and d['logged_date']:
        try:
            from datetime import datetime as dt2
            new_date = dt2.strptime(d['logged_date'], '%Y-%m-%d')
            orig = meal.logged_at
            meal.logged_at = orig.replace(year=new_date.year, month=new_date.month, day=new_date.day)
        except ValueError:
            return jsonify({'error': 'Invalid date format'}), 400

    db.session.commit()
    return jsonify({
        'success':   True,
        'food_name': meal.food_name,
        'calories':  meal.calories,
        'protein':   meal.protein,
        'carbs':     meal.carbs,
        'fat':       meal.fat,
        'fiber':     meal.fiber,
        'serving_g': meal.serving_g,
        'meal_type': meal.meal_type,
    })

@app.route('/log')
@login_required
def log():
    from datetime import datetime as dt
    date_str = request.args.get('date', '')
    my_today = malaysia_today()
    try:
        view_date = dt.strptime(date_str, '%Y-%m-%d').date() if date_str else my_today
    except ValueError:
        view_date = my_today

    meals = (MealLog.query
             .filter_by(user_id=current_user.id)
             .filter(db.func.date(MealLog.logged_at) == view_date)
             .order_by(MealLog.logged_at)
             .all())
    totals = {k: round(sum(getattr(m, k) for m in meals), 1)
              for k in ('calories', 'protein', 'carbs', 'fat', 'fiber')}
    food_names = sorted(NUTRITION['food_name_display'].tolist())
    is_today  = (view_date == my_today)
    prev_date = (view_date - timedelta(days=1)).isoformat()
    next_date = (view_date + timedelta(days=1)).isoformat()

    return render_template('log.html',
        meals=meals,
        totals=totals,
        food_names=food_names,
        view_date=view_date.isoformat(),
        today_iso=my_today.isoformat(),
        today_label=view_date.strftime('%A, %d %b %Y'),
        is_today=is_today,
        prev_date=prev_date,
        next_date=next_date,
    )

@app.route('/dashboard')
@login_required
def dashboard():
    my_today = malaysia_today()
    end = my_today; start = end - timedelta(days=6)
    meals = MealLog.query.filter_by(user_id=current_user.id).filter(db.func.date(MealLog.logged_at) >= start).all()
    days = []
    for i in range(7):
        d = start + timedelta(days=i); dm = [m for m in meals if m.logged_at.date() == d]
        days.append({'label': d.strftime('%a'), 'date': d.strftime('%d %b'),
                     'calories': round(sum(m.calories for m in dm), 1),
                     'protein':  round(sum(m.protein  for m in dm), 1),
                     'carbs':    round(sum(m.carbs    for m in dm), 1),
                     'fat':      round(sum(m.fat      for m in dm), 1),
                     'count':    len(dm)})
    wt = {'calories': round(sum(m.calories for m in meals), 1),
          'protein':  round(sum(m.protein  for m in meals), 1),
          'carbs':    round(sum(m.carbs    for m in meals), 1),
          'fat':      round(sum(m.fat      for m in meals), 1),
          'meals':    len(meals), 'days_logged': sum(1 for d in days if d['count'] > 0)}
    return render_template('dashboard.html', days=days, totals=wt)

@app.route('/food_labels')
@login_required
def food_labels():
    labels = [
        {'label': label, 'name': str(NUTRITION.loc[label, 'food_name_display'])}
        for label in NUTRITION.index
    ]
    labels.sort(key=lambda x: x['name'])
    return jsonify(labels)

@app.route('/nutrition/<food_label>')
@login_required
def nutrition_lookup(food_label):
    if food_label not in NUTRITION.index: return jsonify({'error': 'Not found'}), 404
    row = NUTRITION.loc[food_label]; s = float(row['serving_size_g']) / 100
    return jsonify({'food_label': food_label, 'food_name': str(row['food_name_display']),
                    'serving_g': float(row['serving_size_g']),
                    'calories': round(float(row['calories_per_100g']) * s, 1),
                    'protein':  round(float(row['protein_per_100g'])  * s, 1),
                    'carbs':    round(float(row['carbs_per_100g'])    * s, 1),
                    'fat':      round(float(row['fat_per_100g'])      * s, 1),
                    'fiber':    round(float(row['fiber_per_100g'])    * s, 1)})

@app.route('/dietary_advice')
@login_required
def dietary_advice():
    my_today = malaysia_today()
    meals = MealLog.query.filter_by(user_id=current_user.id).filter(
        db.func.date(MealLog.logged_at) == my_today
    ).order_by(MealLog.logged_at).all()

    if not meals:
        return jsonify({'advice': 'No meals logged today yet. Start by scanning a meal to get personalised advice!'})

    cal  = round(sum(m.calories for m in meals), 1)
    pro  = round(sum(m.protein  for m in meals), 1)
    carb = round(sum(m.carbs    for m in meals), 1)
    fat  = round(sum(m.fat      for m in meals), 1)
    fib  = round(sum(m.fiber    for m in meals), 1)
    n    = len(meals)

    tips = []

    if cal < 500:
        tips.append(f"You have only consumed {cal} kcal today — that is quite low. Make sure to eat enough to fuel your body throughout the day.")
    elif cal < 1200:
        tips.append(f"Your intake is {cal} kcal so far. You still have plenty of room for nutritious meals — aim for around 2000 kcal for the day.")
    elif cal < 1600:
        tips.append(f"You are at {cal} kcal for the day, which is a good moderate level. Keep going with balanced meals.")
    elif cal <= 2200:
        tips.append(f"Your calorie intake of {cal} kcal is well within a healthy range for most adults. Great balance today!")
    elif cal <= 2800:
        tips.append(f"You have reached {cal} kcal today, which is slightly above the typical 2000 kcal target. Consider lighter options for your remaining meals.")
    else:
        tips.append(f"You have consumed {cal} kcal today, which is significantly above the recommended daily intake. Try to choose lighter, lower-calorie options for the rest of the day.")

    if pro < 30:
        tips.append(f"Your protein intake is low at {pro}g. Try adding a protein-rich food like eggs, chicken, tofu, or legumes to your next meal.")
    elif pro < 50:
        tips.append(f"You have had {pro}g of protein so far. Adding one more protein source such as fish, chicken rice, or a boiled egg would help you reach the recommended 50-60g daily.")
    else:
        tips.append(f"Good job on your protein intake of {pro}g — you are well on track to meet your daily protein needs.")

    if carb > 300:
        tips.append(f"Your carbohydrate intake is high at {carb}g. Consider balancing with more vegetables and protein in your next meal.")
    elif carb > 200:
        tips.append(f"Carbs are at {carb}g — reasonable, but watch out if you plan on more rice or noodle-based dishes later.")

    if fib < 10:
        tips.append(f"Your fibre intake is only {fib}g. Adding vegetables, fruits, or whole grains to your meals would help you reach the recommended 25-30g daily target.")
    elif fib < 20:
        tips.append(f"Fibre is at {fib}g — decent, but try adding some vegetables or fruit to your next meal to boost it further.")
    else:
        tips.append(f"Excellent fibre intake of {fib}g today — your digestive health will thank you!")

    if fat > 80:
        tips.append(f"Fat intake is high at {fat}g. Consider grilled or steamed options rather than fried foods for your remaining meals.")

    if n == 1:
        tips.append("You have only logged one meal today. Regular meals help maintain steady energy levels and prevent overeating later.")

    advice = " ".join(tips)
    return jsonify({'advice': advice})


# runs under both gunicorn and direct python
with app.app_context():
    db.create_all()
    print("Database ready.")

preload_all_models()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)