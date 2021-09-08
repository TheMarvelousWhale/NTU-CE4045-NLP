package ntu.nlp.aos

import android.util.Log
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import ntu.nlp.aos.api.ApiConstants
import ntu.nlp.aos.api.NlpApi
import org.json.JSONException
import org.json.JSONObject
import retrofit2.HttpException
import java.lang.Exception


private const val TAG = "NLP.Repo"
object Repository {
    val results = MutableLiveData<MutableList<String>>()

    suspend fun generateText(prompt: String) {
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "generateText start")
                val jsonResult = NlpApi.retrofitService.generateTextAsync(prompt).await()
                Log.d(TAG, "generateText end")
                Log.d(TAG, "result=$jsonResult")

                // TODO: parse and pass
                val resultList = mutableListOf<String>()
                try{
                    val jObject = JSONObject(jsonResult)
                    val suggestions = jObject.getJSONArray(ApiConstants.KEY_RESULT)
                    for (i in 0 until suggestions.length()) {
                        val value = suggestions.getString(i)
                        Log.d(TAG, "$i, $value")
                        resultList.add(value)
                    }
                }catch (e: Exception){
                    e.printStackTrace()
                }

                // post result on UI thread
                results.postValue(resultList)
            } catch (he: HttpException) {
                Log.e(TAG, "generateText.err.http = ${he.message}", he)
                he.printStackTrace()
            } catch (e: Exception){
                Log.e(TAG, "generateText.err = ${e.message}", e)
                e.printStackTrace()
            }
        }
    }



}