package ntu.nlp.aos

import android.util.Log
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import ntu.nlp.aos.api.ApiConstants
import ntu.nlp.aos.api.NlpApi
import ntu.nlp.aos.entity.ResponseResult
import org.json.JSONObject
import retrofit2.HttpException
import java.lang.Exception


private const val TAG = "NLP.Repo"
object Repository {
    val result = MutableLiveData<ResponseResult>()

    suspend fun generateText(prompt: String, num_reviews: Int) {
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "generateText start")
                val timeStart = System.currentTimeMillis()
                val jsonResult = NlpApi.retrofitService.generateTextAsync(prompt, num_reviews).await()
                val timeTaken = System.currentTimeMillis() - timeStart
                Log.d(TAG, "generateText end")
                Log.d(TAG, "result=$jsonResult")


                // parsing the json
                try{
                    val parseResult = parseTextGenerationResult(prompt, jsonResult, timeTaken)

                    // post result on UI thread
                    result.postValue(parseResult)
                }catch (e: Exception){
                    e.printStackTrace()
                }

            } catch (he: HttpException) {
                Log.e(TAG, "generateText.err.http = ${he.message}", he)
                he.printStackTrace()
            } catch (e: Exception){
                Log.e(TAG, "generateText.err = ${e.message}", e)
                e.printStackTrace()
            }
        }
    }

    // custom parsing, so will not affected a lot by change in format
    private fun parseTextGenerationResult(prompt: String, json: String, timeTaken: Long): ResponseResult {
        val result = ResponseResult(prompt = prompt, totalTime = timeTaken)
        val jObject = JSONObject(json)
        val suggestions = jObject.getJSONArray(ApiConstants.KEY_RESULT)
        for (i in 0 until suggestions.length()) {
            val value = suggestions.getString(i)
            Log.d(TAG, "$i, $value")
            result.results?.add(value)
        }
        return result
    }



}