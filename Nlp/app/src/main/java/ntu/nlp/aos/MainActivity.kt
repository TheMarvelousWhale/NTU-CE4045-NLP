package ntu.nlp.aos

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.viewModels
import ntu.nlp.aos.adapter.BaseClickedListener
import ntu.nlp.aos.databinding.ActivityMainBinding
import ntu.nlp.aos.util.copyToClipboard


private const val TAG = "NLP.MainAct"
class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    private val _viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)

        // init
        initBinding()
        observeViewModel()
        setContentView(binding.root)
    }

    private fun initBinding(){
        with(binding){
            // init binding
            viewModel = _viewModel
            lifecycleOwner = this@MainActivity
            // some other binding
            rvResults.adapter = _viewModel.adapter
        }
    }

    private fun observeViewModel(){
        _viewModel.adapter.clickedListener = BaseClickedListener { action, viewHolder ->
            val text = viewHolder.itemView.tag as String
            copyToClipboard(text)
            Toast.makeText(this, "'$text' copy to copyToClipboard", Toast.LENGTH_SHORT).show()
        }
    }

}