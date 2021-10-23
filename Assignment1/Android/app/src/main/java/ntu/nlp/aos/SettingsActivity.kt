package ntu.nlp.aos

import android.os.Bundle
import android.view.MenuItem
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.PreferenceManager


class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.settings_activity)
        if (savedInstanceState == null) {
            supportFragmentManager
                .beginTransaction()
                .replace(R.id.settings, SettingsFragment())
                .commit()
        }
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }


    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            android.R.id.home -> {
                onBackPressed()
                return true
            }
        }
        return super.onOptionsItemSelected(item)
    }

    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            val manager: PreferenceManager = preferenceManager
            manager.sharedPreferencesName = getString(R.string.pref_name)
            setPreferencesFromResource(R.xml.root_preferences, rootKey)

            // allow change theme color
            val keyTheme = getString(R.string.key_app_theme)
            findPreference<Preference>(keyTheme)?.onPreferenceClickListener = Preference.OnPreferenceClickListener {
                val title = getString(R.string.title_app_theme)
                val options = resources.getStringArray(R.array.theme_options)
                AlertDialog.Builder(requireContext())
                    .setTitle(title)
                    .setItems(options){dialog, which->
                        when (which){
                            1 -> AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)
                            2 -> AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
                            else -> AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM)
                        }
                    }
                    .create()
                    .show()
                true
            }
        }
    }
}
