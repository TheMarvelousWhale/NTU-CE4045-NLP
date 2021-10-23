package ntu.nlp.aos.util

import android.text.Editable
import android.text.TextWatcher
import android.widget.EditText
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context

import androidx.core.content.ContextCompat





//fun EditText.afterTextChanged(afterTextChanged: (String) -> Unit) {
//    this.addTextChangedListener(object : TextWatcher {
//        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
//        }
//
//        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
//        }
//
//        override fun afterTextChanged(editable: Editable?) {
//            afterTextChanged.invoke(editable.toString())
//        }
//    })
//}


fun Context.copyToClipboard(text:String){
    val clipboard: ClipboardManager? = ContextCompat.getSystemService(this, ClipboardManager::class.java)
    val clip = ClipData.newPlainText("Copied Text", text)
    clipboard?.setPrimaryClip(clip)
}