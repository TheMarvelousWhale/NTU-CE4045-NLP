package ntu.nlp.aos.adapter

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.databinding.ViewDataBinding
import androidx.recyclerview.widget.RecyclerView
import ntu.nlp.aos.BR
import ntu.nlp.aos.databinding.CardResultBinding

class SimpleCardAdapter: RecyclerView.Adapter<SimpleCardAdapter.CardViewHolder>(){
    companion object{
        const val ACTION_ITEM_CLICK = 1001
    }
    var items: List<String>? = mutableListOf()
    var clickedListener: BaseClickedListener<CardViewHolder>? = null

    override fun getItemCount() = items?.size ?: 0

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): CardViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        val binding = CardResultBinding.inflate(inflater, parent, false)
        return CardViewHolder(binding)
    }

    override fun onBindViewHolder(holder: CardViewHolder, position: Int) {
        holder.bindAs(items!![position], clickedListener)
    }



    // custom view holder
    class CardViewHolder(val binding: ViewDataBinding) : RecyclerView.ViewHolder(binding.root){
        fun bindAs(text: String, clickedListener: BaseClickedListener<CardViewHolder>?){
            binding.setVariable(BR.item, text)
            itemView.tag = text
            itemView.setOnClickListener{
                clickedListener?.onClick(ACTION_ITEM_CLICK, this)
            }
        }
    }
}