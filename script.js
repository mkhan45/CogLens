function search(event) {
    code = (event.keyCode? event.keyCode : event.which);
    entry = "";
    let input = document.getElementById("bar").value;
    entry += input;
    console.log(input);
    if (code == 13) {
        return_val = entry;
        entry = "";
        console.log(return_val);
        return return_val
    }
}