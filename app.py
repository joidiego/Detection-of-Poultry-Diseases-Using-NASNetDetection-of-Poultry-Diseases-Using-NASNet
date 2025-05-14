from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
MODEL_PATH = 'Nasnet_finetuned_model.h5'

# Label sesuai urutan dataset saat training
CLASS_NAMES = ['Coccidiosis', 'healthy', 'ncd', 'salmo']

# Load model
model = load_model(MODEL_PATH, compile=False)

def predict_image(img_path, threshold=75):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    confidence = round(100 * np.max(predictions), 2)
    
    if confidence < threshold:
        return "Data tidak benar", confidence
    else:
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'gambar' not in request.files:
            return redirect(request.url)

        file = request.files['gambar']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predicted_class, confidence = predict_image(filepath)
            
            uploaded_image = url_for('static', filename='uploads/' + filename)
            
            return render_template('result.html',
                                   uploaded_image=uploaded_image,
                                   prediction=predicted_class,
                                   confidence=confidence)
    
    return render_template('predict.html')

@app.route('/info')
def info_penyakit():
    penyakit_list = [
        {
            "nama": "1. Cocci (Coccidiosis)",
            "deskripsi": "Coccidiosis disebabkan oleh parasit mikroskopis bernama Eimeria yang menyerang usus ayam. Parasit ini berkembang biak dengan cepat di lingkungan yang lembap, kotor, dan padat, terutama saat sanitasi kandang tidak dijaga dengan baik. Ayam muda sangat rentan terhadap infeksi ini karena sistem kekebalan mereka belum sepenuhnya berkembang..",
            "gejala": ["Kotoran berdarah atau berlendir", "Nafsu makan menurun", "Bulu kusam dan ayam lesu", "Pertumbuhan lambat dan sering dehidrasi" ],
            "pengobatan": "Penanganan coccidiosis dilakukan dengan pemberian obat anticoccidia seperti Amprolium atau Sulfaquinoxaline yang dicampurkan dalam air minum. Selain itu, sangat penting untuk menjaga kebersihan kandang, mengontrol kelembapan, dan mencegah penumpukan kotoran yang menjadi tempat berkembangnya parasit. Pemberian vitamin, terutama vitamin A dan K, juga dapat membantu mempercepat pemulihan ayam. Untuk pencegahan jangka panjang, vaksinasi dan manajemen biosekuriti yang ketat sangat dianjurkan."
        },
        {
            "nama": "2. NCD (Newcastle Disease / Tetelo)",
            "deskripsi": "Newcastle Disease adalah penyakit menular yang disebabkan oleh virus dari keluarga Paramyxovirus. Virus ini sangat agresif dan mudah menyebar antar ayam melalui udara, air minum, pakan, bahkan peralatan atau manusia yang terkontaminasi. NCD dapat menyerang sistem pernapasan, saraf, dan pencernaan ayam secara bersamaan..",
            "gejala": ["Sesak napas, bersin, suara serak.", "Leher terpuntir (torticollis).", "Diare hijau atau putih.", "Penurunan produksi telur, kelumpuhan.", "Kematian mendadak dalam jumlah besar."],
            "pengobatan": "Karena tidak ada obat untuk Newcastle Disease, langkah terbaik adalah melalui pencegahan. Vaksinasi merupakan cara utama untuk melindungi ayam dari virus ini dan harus dilakukan secara rutin mulai dari usia dini. Jika terjadi infeksi, ayam yang sakit perlu segera diisolasi untuk mencegah penyebaran lebih lanjut. Kandang dan peralatan harus didesinfeksi secara menyeluruh, dan lalu lintas manusia serta hewan ke dalam kandang harus dibatasi selama masa wabah.."
        },
        {
            "nama": "3. Salmo (Salmonellosis)",
            "deskripsi": "Salmonellosis disebabkan oleh infeksi bakteri Salmonella, yang biasanya masuk ke tubuh ayam melalui makanan, air, atau lingkungan kandang yang tercemar. Penyakit ini sangat berbahaya, terutama bagi anak ayam, dan dapat menyebar dengan cepat jika kebersihan kandang tidak dijaga dengan baik.",
            "gejala": ["Diare berbau tajam, kotoran kehijauan atau putih.", "Nafsu makan turun.", "Lesu, dehidrasi." ,"Kadang menyebabkan kematian mendadak pada anak ayam."],
            "pengobatan": "Salmonellosis dapat diatasi dengan pemberian antibiotik seperti Enrofloxacin, Neomycin, atau Tetracycline sesuai dosis dan anjuran dokter hewan. Selain itu, sanitasi yang baik menjadi kunci utama dalam pencegahan, termasuk pembersihan rutin kandang, peralatan, serta pengendalian hama seperti tikus dan serangga. Setelah pengobatan, pemberian probiotik dan multivitamin sangat disarankan untuk membantu pemulihan saluran pencernaan ayam dan meningkatkan daya tahan tubuhnya."
        },
        {
            "nama": "4. Gumboro (Infectious Bursal Disease / IBD)",
            "deskripsi": "Gumboro disebabkan oleh virus yang menyerang bursa Fabricius, yaitu organ penting dalam sistem kekebalan ayam. Virus ini sangat menular, terutama menyerang ayam berumur 3 hingga 6 minggu. Penularan terjadi melalui kontak langsung, pakan, air minum, dan peralatan yang terkontaminasi. Virus Gumboro sangat tahan terhadap lingkungan dan bisa bertahan lama di kandang yang tidak disanitasi dengan baik..",
            "gejala": ["Ayam terlihat lesu dan enggan bergerak.", "Bulu kusut dan berdiri.", "Diare berwarna putih atau encer.", "Penurunan nafsu makan dan dehidrasi.", "Angka kematian bisa tinggi dalam waktu singkat."],
            "pengobatan": "Pencegahan Gumboro dilakukan melalui vaksinasi rutin, biasanya dimulai saat ayam berumur 2 minggu. Vaksin harus disimpan dan diberikan sesuai prosedur agar efektif. Selain itu, sanitasi kandang harus dijaga dengan baik, termasuk desinfeksi berkala dan kontrol kepadatan ayam untuk menghindari penyebaran virus."
        },
        {
            "nama": "5. CRD (Chronic Respiratory Disease)y",
            "deskripsi": "CRD disebabkan oleh infeksi bakteri Mycoplasma gallisepticum yang menyerang saluran pernapasan ayam. Penyakit ini cenderung berkembang dalam kondisi kandang yang buruk, ventilasi tidak memadai, atau stres akibat perubahan cuaca. Penularannya dapat terjadi melalui udara, air, serta kontak langsung antar ayam.",
            "gejala": ["Batuk, bersin, dan suara napas yang kasar.", "Keluar lendir dari hidung atau mata.", "Nafsu makan menurun dan pertumbuhan lambat.", "Sayap turun, ayam tampak lelah dan mengantuk.", "Penurunan produksi telur pada ayam petelur."],
            "pengobatan": "Pencegahan CRD dapat dilakukan dengan menjaga kualitas ventilasi kandang agar udara tetap bersih dan segar. Hindari kepadatan yang berlebihan serta stres pada ayam. Vaksinasi terhadap Mycoplasma bisa diberikan di peternakan intensif, dan kebersihan lingkungan harus selalu dijaga. Peternak juga harus membatasi lalu lintas manusia atau alat yang keluar masuk kandang."
        },
        {
            "nama": "6. Avian Influenza (Flu Burung)",
            "deskripsi": "Flu burung disebabkan oleh virus influenza tipe A, terutama subtipe H5 dan H7, yang sangat menular dan mematikan. Virus ini bisa menyebar melalui udara, kotoran, air minum, peralatan kandang, serta burung liar. Penyakit ini tergolong zoonosis, artinya bisa menular ke manusia dalam kondisi tertentu.",
            "gejala": ["Kematian mendadak dalam jumlah besar tanpa gejala awal.", "Ayam tampak lesu, sulit bernapas, dan tidak aktif.", "Pembengkakan di area wajah, jengger, dan pial.", "Warna kebiruan pada jengger, pial, dan kaki.", "Penurunan drastis dalam produksi telur."],
            "pengobatan": "Karena tidak ada obat untuk flu burung, pencegahan menjadi langkah utama. Peternakan harus menerapkan biosekuriti ketat: membatasi akses ke kandang, mencegah kontak dengan burung liar, menyaring pakan dan air, serta mendesinfeksi peralatan secara berkala. Vaksin flu burung tersedia untuk beberapa jenis, tetapi penggunaannya harus diatur oleh otoritas kesehatan hewan. Jika terjadi wabah, ayam yang terinfeksi harus segera dimusnahkan dan lokasi dikarantina."
        }
    ]
    return render_template('info_penyakit.html', penyakit_list=penyakit_list)
if __name__ == "__main__":
    app.run(debug=True)
