package ntu.nlp.aos.adapter

import androidx.recyclerview.widget.RecyclerView

fun interface BaseClickedListener<VH: RecyclerView.ViewHolder>{
    fun onClick(action: Int, viewHolder: VH)
}
