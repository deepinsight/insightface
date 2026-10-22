package com.example.inspireface_example;

import android.content.Context;
import android.os.Build;

import androidx.appcompat.app.AppCompatDelegate;
import androidx.core.os.LocaleListCompat;

import java.util.Locale;

/**
 * Per-app language selection. The demo defaults to English regardless of the system
 * language; the in-app language controls switch to Chinese and back.
 * The choice is stored in SharedPreferences and re-applied on every app start.
 */
public final class LocalePrefs {

    private static final String PREFS = "settings";
    private static final String KEY_LOCALE = "app_locale";
    private static final String ENGLISH = "en";
    private static final String CHINESE = "zh";

    private LocalePrefs() {
    }

    /** Applies the stored language (English by default). Call from Application.onCreate. */
    public static void applyStored(Context context) {
        String selected = stored(context);
        if (Build.VERSION.SDK_INT >= 33) {
            String platformSelection = supportedLanguage(
                    AppCompatDelegate.getApplicationLocales());
            if (platformSelection != null) {
                // Respect a language selected from Android 13+'s per-app language screen.
                selected = platformSelection;
            }
        }
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit().putString(KEY_LOCALE, selected).apply();
        AppCompatDelegate.setApplicationLocales(
                LocaleListCompat.forLanguageTags(selected));
    }

    /** Switches between English and Chinese; running activities recreate automatically. */
    public static void toggle(Context context) {
        String active = supportedLanguage(AppCompatDelegate.getApplicationLocales());
        String next = CHINESE.equals(active != null ? active : stored(context))
                ? ENGLISH : CHINESE;
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit().putString(KEY_LOCALE, next).apply();
        AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(next));
    }

    private static String stored(Context context) {
        String value = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString(KEY_LOCALE, ENGLISH);
        return CHINESE.equals(value) ? CHINESE : ENGLISH;
    }

    private static String supportedLanguage(LocaleListCompat locales) {
        if (locales == null || locales.isEmpty()) {
            return null;
        }
        Locale locale = locales.get(0);
        if (locale == null) {
            return null;
        }
        if (CHINESE.equals(locale.getLanguage())) {
            return CHINESE;
        }
        return ENGLISH.equals(locale.getLanguage()) ? ENGLISH : null;
    }
}
