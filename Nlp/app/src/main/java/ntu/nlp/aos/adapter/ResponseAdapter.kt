package ntu.nlp.aos.adapter

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import ntu.nlp.aos.BR
import ntu.nlp.aos.databinding.CardHeaderBinding
import ntu.nlp.aos.databinding.CardResultBinding
import ntu.nlp.aos.entity.ResponseResult

class ResponseAdapter: RecyclerView.Adapter<BaseRecyclerViewHolder>(){
    companion object{
        private const val VIEW_TYPE_HEADER = 1
        private const val VIEW_TYPE_ITEM = 2
        const val ACTION_ITEM_CLICK = 1001
    }
    var responseResult: ResponseResult? = null
    var clickedListener: BaseClickedListener<BaseRecyclerViewHolder>? = null

    override fun getItemCount() = when{
        responseResult == null -> 0
        else -> responseResult!!.results!!.size + 1
    }

    override fun getItemViewType(position: Int) = when(position) {
        0 -> VIEW_TYPE_HEADER
        else -> VIEW_TYPE_ITEM
    }


    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): BaseRecyclerViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return when(viewType){
            VIEW_TYPE_HEADER -> {
                val binding = CardHeaderBinding.inflate(inflater, parent, false)
                HeaderCardViewHolder(binding)
            }
            else -> {
                val binding = CardResultBinding.inflate(inflater, parent, false)
                ItemCardViewHolder(binding)
            }
        }
    }

    override fun onBindViewHolder(holder: BaseRecyclerViewHolder, position: Int) = when(position){
        0 -> (holder as HeaderCardViewHolder).bindAs(responseResult!!)
        else -> (holder as ItemCardViewHolder).bindAs(responseResult!!.results!!.get(position-1), clickedListener)
    }




    // custom view holder
    class HeaderCardViewHolder(override val binding: CardHeaderBinding) : BaseRecyclerViewHolder(binding){

        fun bindAs(result: ResponseResult){
            binding.setVariable(BR.result, result)
            itemView.tag = result
        }
    }

    class ItemCardViewHolder(override val binding: CardResultBinding) : BaseRecyclerViewHolder(binding){
        fun bindAs(text: String, clickedListener: BaseClickedListener<BaseRecyclerViewHolder>?){
            binding.setVariable(BR.item, text)
            itemView.tag = text
            itemView.setOnClickListener{
                clickedListener?.onClick(ACTION_ITEM_CLICK, this)
            }
        }
    }
}