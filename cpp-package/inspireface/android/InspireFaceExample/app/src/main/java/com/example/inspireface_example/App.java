package com.example.inspireface_example;

import android.app.Application;

public class App extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        LocalePrefs.applyStored(this);
    }
}
