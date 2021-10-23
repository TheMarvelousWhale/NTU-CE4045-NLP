package ntu.nlp.aos

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Toast
import androidx.activity.viewModels
import ntu.nlp.aos.adapter.BaseClickedListener
import ntu.nlp.aos.databinding.ActivityMainBinding
import ntu.nlp.aos.util.copyToClipboard
import android.util.Log
import android.view.Menu
import android.view.MenuItem


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
            Toast.makeText(this, "copied to copyToClipboard", Toast.LENGTH_SHORT).show()
        }

        _viewModel.result.observe(this, {
            Log.d(TAG, it.toString())


            _viewModel.adapter.apply {
                responseResult = it
                notifyDataSetChanged()
            }
        })
    }


    // option menu
    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        val inflater = menuInflater
        inflater.inflate(R.menu.menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when(item.itemId){
            R.id.action_to_setting -> {
                val intent = Intent(this, SettingsActivity::class.java)
                startActivity(intent)
                true
            }
            else -> super.onContextItemSelected(item)
        }
    }

}