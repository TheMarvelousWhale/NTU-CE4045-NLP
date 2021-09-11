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
        Log.d(TAG, "onSend")
        viewModelScope.launch {

            status.value = ApiStatus.LOADING
            try{
                val pref = getSharePreferences()
                val numReviews = Integer.parseInt(pref.getString("num_reviews", "1"))
                val prompt = input.value.toString()

                Repository.generateText(prompt, numReviews)
                status.value = ApiStatus.DONE
                Log.d(TAG, "onSend complete")
            } catch (e: Exception){
                e.printStackTrace()
                status.value = ApiStatus.ERROR
            }
        }
    }


    private fun getSharePreferences(): SharedPreferences {
        val application = getApplication<Application>()
        val perfName: String = application.getString(R.string.pref_name)
        return application.getSharedPreferences(perfName, Context.MODE_PRIVATE)
    }

}