package ntu.nlp.aos.adapter

import androidx.databinding.ViewDataBinding
import androidx.recyclerview.widget.RecyclerView

open class BaseRecyclerViewHolder (open val binding: ViewDataBinding): RecyclerView.ViewHolder(binding.root)
