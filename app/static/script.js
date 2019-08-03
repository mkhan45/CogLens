function test() {
    alert('Yeah!')
}
function wait() {
    setTimeout(function() {}, 1000);
}
function startRecord() {
    document.getElementById('record').style.visibility = "visible";
    seconds = document.getElementById('time_bar').value;
    setTimeout(show_ok, 1000*seconds + 4000)
}