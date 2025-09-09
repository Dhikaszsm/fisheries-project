from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fisheries-production-key-railway')

# Decorator untuk proteksi role
def require_role(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash('Silakan login terlebih dahulu!')
                return redirect(url_for('login'))
            
            user_role = session.get('role')
            if user_role != required_role:
                flash(f'Akses ditolak! Anda tidak memiliki izin untuk mengakses dashboard {required_role.title()}.')
                return redirect(url_for('welcome'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Simple authentication untuk demo
        demo_users = {
            'user_budidaya': {'password': 'passwordbud', 'role': 'budidaya'},
            'user_tangkap': {'password': 'passwordtang', 'role': 'tangkap'},
            'user_pds': {'password': 'passwordpds', 'role': 'pdspkp'}
        }
        
        if username in demo_users and demo_users[username]['password'] == password:
            session['user_id'] = username
            session['username'] = username
            session['role'] = demo_users[username]['role']
            
            flash(f'Login berhasil! Selamat datang, {username} ({demo_users[username]["role"]}).')
            return redirect(url_for('welcome'))
        else:
            flash('Username atau password salah!')
    
    return render_template('login.html')

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('welcome.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    role = session.get('role')
    
    # Redirect to role-specific dashboard
    if role == 'budidaya':
        return redirect(url_for('dashboard_budidaya'))
    elif role == 'tangkap':
        return redirect(url_for('dashboard_tangkap'))
    elif role == 'pdspkp':
        return redirect(url_for('dashboard_pdspkp'))
    else:
        flash('Role tidak dikenali!')
        return redirect(url_for('welcome'))

@app.route('/dashboard/budidaya')
@require_role('budidaya')
def dashboard_budidaya():
    # Sample data untuk demo
    stats = {
        'total_kolam': 125,
        'produksi_bulan': 2.5,
        'kualitas_air': 'Good',
        'alerts': 3
    }
    
    return render_template('dashboard_budidaya.html', stats=stats)

@app.route('/dashboard/tangkap')
@require_role('tangkap')
def dashboard_tangkap():
    return render_template('dashboard_tangkap.html')

@app.route('/dashboard/pdspkp')
@require_role('pdspkp')
def dashboard_pdspkp():
    return render_template('dashboard_pdspkp.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah logout.')
    return redirect(url_for('login'))

# Face recognition placeholders (disabled untuk testing)
@app.route('/face/enrollment')
def face_enrollment_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    flash('Face Recognition sedang dalam maintenance')
    return redirect(url_for('welcome'))

@app.route('/face/login')
def face_login_page():
    flash('Face Recognition sedang dalam maintenance')
    return redirect(url_for('login'))

if __name__ == '__main__':
    print("=== Fisheries System - Simple Mode ===")
    print("Face Recognition: DISABLED")  
    print("Redis: DISABLED")
    print("URL: http://localhost:5000")
    print("=" * 40)
    print("Login accounts:")
    print("- user_budidaya / passwordbud")
    print("- user_tangkap / passwordtang") 
    print("- user_pds / passwordpds")
    print("=" * 40)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
