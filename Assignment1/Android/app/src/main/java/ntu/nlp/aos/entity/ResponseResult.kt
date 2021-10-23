package ntu.nlp.aos.entity

import java.util.concurrent.TimeUnit


data class ResponseResult(
    val prompt: String?= null,
    val results: MutableList<String>?= mutableListOf(),
    val queryTime: Double?= 0.0,
    val totalTime: Long,
    val timestamp: Double?= 0.0
){
    fun formatTime(): String{
        val mins = TimeUnit.MILLISECONDS.toMinutes(totalTime)
        val sec  = TimeUnit.MILLISECONDS.toSeconds(totalTime)
        return when{
            mins>0 -> "$mins mins $sec s"
            else -> "$sec s"
        }
    }
}