package ntu.nlp.aos.util

import android.util.Log
import androidx.databinding.BindingAdapter
import androidx.lifecycle.MutableLiveData
import androidx.recyclerview.widget.RecyclerView
import ntu.nlp.aos.adapter.SimpleCardAdapter

private const val TAG = "NLP.BindAdapter"
@BindingAdapter( "simpleAdapter")
fun <T : Any> setSimpleAdapter(recyclerView: RecyclerView, items: MutableLiveData<MutableList<String>>?) {
    Log.d(TAG, "simpleAdapter")
    items?.value?.let { itemList ->
        (recyclerView.adapter as? SimpleCardAdapter)?.apply {
            this.items = itemList
            this.notifyDataSetChanged()
        }
    }
}