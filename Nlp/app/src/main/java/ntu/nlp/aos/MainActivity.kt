package ntu.nlp.aos

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import androidx.activity.viewModels
import ntu.nlp.aos.databinding.ActivityMainBinding


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
            binding.viewModel = _viewModel
            binding.lifecycleOwner = this@MainActivity
            // some other binding
            rgStars.setOnCheckedChangeListener { _, btnId ->
                _viewModel.stars.value = when (btnId){
                    binding.rbStar5.id -> 5
                    binding.rbStar4.id -> 4
                    binding.rbStar3.id -> 3
                    binding.rbStar2.id -> 2
                    else -> 1
                }
            }
        }
    }

    private fun observeViewModel(){
        _viewModel.enableSend.observe(this, { enabled ->
            binding.btnSend.isEnabled = enabled
        })
    }

}