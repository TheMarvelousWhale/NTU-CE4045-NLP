package ntu.nlp.aos


import android.util.Log
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.Transformations
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import java.lang.StringBuilder

private const val TAG = "NLP.MainActVM"
class MainViewModel: ViewModel() {
    val resultList = Repository.results

    val input = MutableLiveData<String>( )
    val enableSend = Transformations.map(input){ text->
        text.isNotEmpty()
    }
    val stars = MutableLiveData(1)
    val useful = MutableLiveData( false )
    val funny = MutableLiveData( false )
    val cool = MutableLiveData( false )

    val result = Transformations.map(resultList){
        val sb = StringBuilder()
        it.forEachIndexed{i, text ->
            sb.append("[$i] $text\n\n")
        }
        sb.toString()
    }



    fun onSend(){
        // TODO: call the API OR model here
        val sb = StringBuilder()
        sb.append("input: ${input.value}\n")
        sb.append("stars: ${stars.value}\n")
        sb.append("useful: ${useful.value}\n")
        sb.append("funny: ${funny.value}\n")
        sb.append("cool: ${cool.value}\n")
        Log.d(TAG, sb.toString())

        viewModelScope.launch {
            Repository.generateText(input.value.toString())
        }

    }

}