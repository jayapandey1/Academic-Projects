package com.example.firstapp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity2 extends AppCompatActivity implements SensorEventListener {
    public static boolean buttonClicked = false;
    public static boolean showAcc = false;
    private SensorManager sensorManager;
    private Sensor sensor;
    private final float[] accelerometerReading = new float[3];
    private final float[] magnetometerReading = new float[3];

    private final float[] rotationMatrix = new float[9];
    private final float[] orientationAngles = new float[3];

    private TextView xText, yText, zText;
    private boolean mInitialized;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        //sensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        //sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL);
        Sensor accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer,
                    SensorManager.SENSOR_DELAY_NORMAL);
        }
        Sensor magneticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        if (magneticField != null) {
            sensorManager.registerListener(this, magneticField,
                    SensorManager.SENSOR_DELAY_NORMAL);
        }

        xText = (TextView) findViewById(R.id.xText);
        Button btnAcc = (Button) findViewById(R.id.btnAcc);
        btnAcc.setOnClickListener(setOnClickListener(this));
        mInitialized = false;
    }

    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.startB:
                sManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL);
                break;
            case R.id.stopB:
                sManager.unregisterListener(this);
                break;
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        xText.setText(" X  :" + Float.toString(event.values[0]) + "\n" +
                " Y  :" + Float.toString(event.values[1]) + "\n" +
                " Z :" + Float.toString(event.values[2]));

        try {
            writeToCsv(Float.toString(event.values[0]), Float.toString(event.values[1]), Float.toString(event.values[2]));
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /*public void writeToCsv(String x, String y, String z) throws IOException {
        Calendar c = Calendar.getInstance();

        File path = Environment.getDataDirectory();
        boolean success = true;
        if (!path.exists()) {
            success = path.mkdir();
        }
        if (success) {
            String csv = "Value.csv";
            FileWriter file_writer = new FileWriter(csv, true);
            String s = c.get(Calendar.HOUR) + "," + c.get(Calendar.MINUTE) + "," + c.get(Calendar.SECOND) + "," + c.get(Calendar.MILLISECOND) + "," + x + "," + y + "," + z + "\n";
            file_writer.append(s);
            file_writer.close();

        }
    }
*/
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    protected void onPause() {
        super.onPause();

        // Don't receive any more updates from either sensor.
        sensorManager.unregisterListener(this);
    }

    // Compute the three orientation angles based on the most recent readings from
    // the device's accelerometer and magnetometer.
    public void updateOrientationAngles() {
        // Update rotation matrix, which is needed to update orientation angles.
        SensorManager.getRotationMatrix(rotationMatrix, null,
                accelerometerReading, magnetometerReading);

        // "rotationMatrix" now has up-to-date information.

        SensorManager.getOrientation(rotationMatrix, orientationAngles);

        // "orientationAngles" now has up-to-date information.
    }
}
