package ntu.nlp.aos


import android.app.Application
import android.content.SharedPreferences
import android.util.Log
import androidx.lifecycle.*
import kotlinx.coroutines.launch
import ntu.nlp.aos.api.ApiStatus
import android.content.Context
import ntu.nlp.aos.adapter.ResponseAdapter


private const val TAG = "NLP.MainActVM"
class MainViewModel(application: Application) : AndroidViewModel(application) {
    val result = Repository.result
    val status = MutableLiveData<ApiStatus>( ApiStatus.IDLE)
    val input = MutableLiveData<String>( )
    val isLoading = Transformations.map(status){ state ->
        ApiStatus.LOADING == state
    }

    val adapter = ResponseAdapter()

    fun onSend(){
        if (input.value.isNullOrEmpty()){
            return
        }
        viewModelScope.launch {
            Log.d(TAG, "request start")
            try{
                status.value = ApiStatus.LOADING
                val pref = getSharePreferences()
                val numReviews = Integer.parseInt(pref.getString("num_reviews", "1"))
                val prompt = input.value.toString()
                input.value = ""

                Repository.generateText(prompt, numReviews)
                status.value = ApiStatus.DONE
            } catch (e: Exception){
                e.printStackTrace()
                status.value = ApiStatus.ERROR
            }
            Log.d(TAG, "request complete")
        }
    }


    private fun getSharePreferences(): SharedPreferences {
        val application = getApplication<Application>()
        val perfName: String = application.getString(R.string.pref_name)
        return application.getSharedPreferences(perfName, Context.MODE_PRIVATE)
    }

}