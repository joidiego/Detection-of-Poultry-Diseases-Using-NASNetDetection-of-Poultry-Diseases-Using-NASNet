document.getElementById('upload').addEventListener('change', function(event) {
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('preview').src = e.target.result;
        document.getElementById('preview').style.display = 'block';
    }
    reader.readAsDataURL(event.target.files[0]);
});