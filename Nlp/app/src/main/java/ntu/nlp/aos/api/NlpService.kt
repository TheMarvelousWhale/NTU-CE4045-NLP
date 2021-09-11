package ntu.nlp.aos.api


import com.jakewharton.retrofit2.adapter.kotlin.coroutines.CoroutineCallAdapterFactory
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import kotlinx.coroutines.Deferred
import ntu.nlp.aos.BuildConfig
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.Response
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import retrofit2.converter.scalars.ScalarsConverterFactory
import retrofit2.http.GET
import retrofit2.http.Query
import java.util.concurrent.TimeUnit

interface NlpService {
    @GET(".")
    fun generateTextAsync(
        @Query(ApiConstants.KEY_REVIEW) prompt: String,
        @Query(ApiConstants.KEY_NUM_REVIEW) num_reviews: Int
    ): Deferred<String>
}

object NlpApi {
    private lateinit var retrofit: Retrofit
    val retrofitService: NlpService by lazy {
        getClient().create(NlpService::class.java)
    }

    private fun getClient(): Retrofit{
        if (!::retrofit.isInitialized) {
            val okHttpClient = OkHttpClient.Builder()
                .addInterceptor { apiKeyInterceptor(it) }
                .connectTimeout(1, TimeUnit.MINUTES)
                .readTimeout(1, TimeUnit.MINUTES)
                .build()

            val moshi = Moshi.Builder()
                .add(KotlinJsonAdapterFactory())
                .build()

            retrofit = Retrofit.Builder()
                .baseUrl(ApiConstants.BASE_URL)
                .client(okHttpClient)
                .addConverterFactory(ScalarsConverterFactory.create())
                .addConverterFactory(MoshiConverterFactory.create(moshi))
                .addCallAdapterFactory(CoroutineCallAdapterFactory())
                .build()
        }
        return retrofit
    }

    private fun apiKeyInterceptor(it: Interceptor.Chain): Response {
        val originalRequest = it.request()
        val originalHttpUrl = originalRequest.url()

        val newHttpUrl = originalHttpUrl.newBuilder()
            .addQueryParameter(ApiConstants.KEY_API_KEY, BuildConfig.NLP_KEY)
            .build()

        val newRequest = originalRequest.newBuilder()
            .url(newHttpUrl)
            .build()

        return it.proceed(newRequest)
    }
}