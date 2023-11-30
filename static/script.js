const dropArea = document.getElementById("drop-area");
const inputfile = document.getElementById("input-file");
const imageView = document.getElementById("img-view");

inputfile.addEventListener("change", uploadImage);

function uploadImage()
{
    let imgLink = URL.createObjectURL(inputfile.files[0]);
    imageView.style.backgroundImage = `url(${imgLink})`;
    imageView.textContent = "";
    imageView.style.border = 0
}
dropArea.addEventListener("dragover",function(e){
    e.preventDefault();
});
dropArea.addEventListener("drop",function(e){
    e.preventDefault();
    inputfile.files = e.dataTransfer.files;
    uploadImage()
});

