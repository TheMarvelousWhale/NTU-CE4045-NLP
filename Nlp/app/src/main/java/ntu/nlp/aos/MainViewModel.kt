package ntu.nlp.aos


import android.util.Log
import androidx.lifecycle.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import ntu.nlp.aos.adapter.SimpleCardAdapter
import ntu.nlp.aos.api.ApiStatus
import java.lang.StringBuilder

private const val TAG = "NLP.MainActVM"
class MainViewModel: ViewModel() {
    val resultList = Repository.results
    val status = MutableLiveData<ApiStatus>( ApiStatus.IDLE)
    val input = MutableLiveData<String>( )
    val isLoading = Transformations.map(status){ state ->
        ApiStatus.LOADING == state
    }

    val adapter = SimpleCardAdapter()

    fun onSend(){
        Log.d(TAG, "onSend")
        viewModelScope.launch {
            status.value = ApiStatus.LOADING
            try{
                Repository.generateText(input.value.toString())
                status.value = ApiStatus.DONE
                Log.d(TAG, "onSend complete")
            } catch (e: Exception){
                e.printStackTrace()
                status.value = ApiStatus.ERROR
            }
        }
    }

}