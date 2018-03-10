#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <EEPROM.h>

/* This driver uses the Adafruit unified sensor library (Adafruit_Sensor),
   which provides a common 'type' for sensor data and some helper functions.

   To use this driver you will also need to download the Adafruit_Sensor
   library and include it in your libraries folder.

   You should also assign a unique ID to this sensor for use with
   the Adafruit Sensor API so that you can identify this particular
   sensor in any data logs, etc.  To assign a unique ID, simply
   provide an appropriate value in the constructor below (12345
   is used by default in this example).

   History
   =======
   2015/MAR/03  - First release (KTOWN)
   2015/AUG/27  - Added calibration and system status helpers
 */

/* Set the delay between fresh samples */
//#define BNO055_SAMPLERATE_DELAY_MS (100)
#define BNO055_SAMPLERATE_DELAY_MS (50)

Adafruit_BNO055 bno = Adafruit_BNO055(55);
float currTime  = 0;
float prevTime = 0;
float dt = 0;
float az = 0;
float vz = 0;

int i = 0;
float z = 0;

double avgAccelZeroZ = 0;
double avgAccelZ = 0;
double temp = 0;
/**************************************************************************/
/*
   Display the raw calibration offset and radius data
 */
/**************************************************************************/
void displaySensorOffsets(const adafruit_bno055_offsets_t &calibData)
{
    Serial.print("Accelerometer: ");
    Serial.print(calibData.accel_offset_x); Serial.print(" ");
    Serial.print(calibData.accel_offset_y); Serial.print(" ");
    Serial.print(calibData.accel_offset_z); Serial.print(" ");

    Serial.print("\nGyro: ");
    Serial.print(calibData.gyro_offset_x); Serial.print(" ");
    Serial.print(calibData.gyro_offset_y); Serial.print(" ");
    Serial.print(calibData.gyro_offset_z); Serial.print(" ");

    Serial.print("\nMag: ");
    Serial.print(calibData.mag_offset_x); Serial.print(" ");
    Serial.print(calibData.mag_offset_y); Serial.print(" ");
    Serial.print(calibData.mag_offset_z); Serial.print(" ");

    Serial.print("\nAccel Radius: ");
    Serial.print(calibData.accel_radius);

    Serial.print("\nMag Radius: ");
    Serial.print(calibData.mag_radius);
}


/**************************************************************************/
/*
   Displays some basic information on this sensor from the unified
   sensor API sensor_t type (see Adafruit_Sensor for more information)
 */
/**************************************************************************/
void displaySensorDetails(void)
{
    sensor_t sensor;
    bno.getSensor(&sensor);
    Serial.println("------------------------------------");
    Serial.print  ("Sensor:       "); Serial.println(sensor.name);
    Serial.print  ("Driver Ver:   "); Serial.println(sensor.version);
    Serial.print  ("Unique ID:    "); Serial.println(sensor.sensor_id);
    Serial.print  ("Max Value:    "); Serial.print(sensor.max_value); Serial.println(" xxx");
    Serial.print  ("Min Value:    "); Serial.print(sensor.min_value); Serial.println(" xxx");
    Serial.print  ("Resolution:   "); Serial.print(sensor.resolution); Serial.println(" xxx");
    Serial.println("------------------------------------");
    Serial.println("");
    delay(500);
}

void displaySensorUnits(void)
{


    //

    //https://learn.adafruit.com/adafruit-bno055-absolute-orientation-sensor/overview


    //  // From doc notes, the units are:
    // https://learn.adafruit.com/adafruit-bno055-absolute-orientation-sensor?view=all
    //    print('Temperature: {} degrees C'.format(sensor.temperature))
    //    print('Accelerometer (m/s^2): {}'.format(sensor.accelerometer))
    //    print('Magnetometer (microteslas): {}'.format(sensor.magnetometer))
    //    print('Gyroscope (deg/sec): {}'.format(sensor.gyroscope))
    //    print('Euler angle: {}'.format(sensor.euler))
    //    print('Quaternion: {}'.format(sensor.quaternion))
    //    print('Linear acceleration (m/s^2): {}'.format(sensor.linear_acceleration))
    //    print('Gravity (m/s^2): {}'.format(sensor.gravity))
    //
    //
    //    temperature - The sensor temperature in degrees Celsius.
    //    accelerometer - This is a 3-tuple of X, Y, Z axis accelerometer values in meters per second squared.
    //    magnetometer - This is a 3-tuple of X, Y, Z axis magnetometer values in microteslas.
    //    gyroscope - This is a 3-tuple of X, Y, Z axis gyroscope values in degrees per second.
    //    euler - This is a 3-tuple of orientation Euler angle values.
    //    quaternion - This is a 4-tuple of orientation quaternion values.
    //    linear_acceleration - This is a 3-tuple of X, Y, Z linear acceleration values (i.e. without effect of gravity) in meters per second squared.
    //    gravity - This is a 3-tuple of X, Y, Z gravity acceleration values (i.e. without the effect of linear acceleration) in meters per second squared.
    //
    // https://learn.adafruit.com/bno055-absolute-orientation-sensor-with-raspberry-pi-and-beaglebone-black/webgl-example#sensor-calibration
}

/**************************************************************************/
/*
   Display some basic info about the sensor status
 */
/**************************************************************************/
void displaySensorStatus(void)
{
    /* Get the system status values (mostly for debugging purposes) */
    uint8_t system_status, self_test_results, system_error;
    system_status = self_test_results = system_error = 0;
    bno.getSystemStatus(&system_status, &self_test_results, &system_error);

    /* Display the results in the Serial Monitor */
    Serial.println("");
    Serial.print("System Status: 0x");
    Serial.println(system_status, HEX);
    Serial.print("Self Test:     0x");
    Serial.println(self_test_results, HEX);
    Serial.print("System Error:  0x");
    Serial.println(system_error, HEX);
    Serial.println("");
    delay(500);
}

/**************************************************************************/
/*
   Display sensor calibration status
 */
/**************************************************************************/
void displayCalStatus(void)
{
    /* Get the four calibration values (0..3) */
    /* Any sensor data reporting 0 should be ignored, */
    /* 3 means 'fully calibrated" */
    uint8_t system, gyro, accel, mag;
    system = gyro = accel = mag = 0;
    bno.getCalibration(&system, &gyro, &accel, &mag);

    /* The data should be ignored until the system calibration is > 0 */
    Serial.print("\t");
    if (!system)
    {
        Serial.print("! ");
    }

    /* Display the individual values */
    Serial.print("Sys:");
    Serial.print(system, DEC);
    Serial.print(" G:");
    Serial.print(gyro, DEC);
    Serial.print(" A:");
    Serial.print(accel, DEC);
    Serial.print(" M:");
    Serial.print(mag, DEC);
}

/**************************************************************************/
/*
   Arduino setup function (automatically called at startup)
 */
/**************************************************************************/
void setup(void)
{
    Serial.begin(115200);
    Serial.println("Orientation Sensor Test"); Serial.println("");

    /* Initialise the sensor */
    if(!bno.begin())
    {
        /* There was a problem detecting the BNO055 ... check your connections */
        Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
        while(1);
    }


    /* Display some basic information on this sensor */
    //displaySensorDetails();

    /* Optional: Display current status */
    //displaySensorStatus();

    //displaySensorUnits();



    sensors_event_t event;
    bno.getEvent(&event);

    int eeAddress = 0;
    long bnoID;
    bool foundCalib = false;

    EEPROM.get(eeAddress, bnoID);

    adafruit_bno055_offsets_t calibrationData;
    sensor_t sensor;

    /*
     *  Look for the sensor's unique ID at the beginning oF EEPROM.
     *  This isn't foolproof, but it's better than nothing.
     */
    bno.getSensor(&sensor);
    if (bnoID != sensor.sensor_id)
    {
        Serial.println("\nNo Calibration Data for this sensor exists in EEPROM");
        delay(500);
    }
    else
    {
        Serial.println("\nFound Calibration for this sensor in EEPROM.");
        eeAddress += sizeof(long);
        EEPROM.get(eeAddress, calibrationData);

        displaySensorOffsets(calibrationData);

        Serial.println("\n\nRestoring Calibration data to the BNO055...");
        bno.setSensorOffsets(calibrationData);

        Serial.println("\n\nCalibration data loaded into BNO055");
        foundCalib = true;

    }

    delay(1000);
    if (foundCalib){
        Serial.println(":] Move sensor slightly to calibrate magnetometers");
        while (!bno.isFullyCalibrated())
        {
            Serial.println("");
            bno.getEvent(&event);
            displayCalStatus();

            delay(BNO055_SAMPLERATE_DELAY_MS);
        }
    }

    Serial.println("\nFully calibrated!");
    Serial.println("--------------------------------");
    Serial.println("Calibration Results: ");
    adafruit_bno055_offsets_t newCalib;
    bno.getSensorOffsets(newCalib);
    displaySensorOffsets(newCalib);


    bno.setExtCrystalUse(true);
}

/**************************************************************************/
/*
   Arduino loop function, called once 'setup' is complete (your own code
   should go here)
 */
/**************************************************************************/
void loop(void) {


    /* Get a new sensor event */
    //sensors_event_t event;
    // bno.getEvent(&event);
    //https://github.com/adafruit/Adafruit_BNO055/blob/5565ed3497994fc74c18e9270ff74e205e8c839b/Adafruit_BNO055.cpp#L337
    imu::Vector<3> grav = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY); // !!!!
    imu::Vector<3> linaccel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL); // !!!!

       Serial.print("gravX: ");
       Serial.print(grav.x(), 4);
       Serial.print("\tgravY: ");
       Serial.print(grav.y(), 4);
       Serial.print("\tgravZ: ");
       Serial.print(grav.z(), 4);
       Serial.println("");

    //temp = linaccel.z() - avgAccelZeroZ; 
/*
    if  (temp < 0.005) { az += 0; }
        else{az += temp;}
    if (i%5 == 0){
        avgAccelZ = az/5;

        currTime = millis();
        dt = (currTime - prevTime) / 1000.0; //seconds
        prevTime = currTime;

        vz += avgAccelZ * dt;
        z += vz * dt / 1000; // millimeters
        az = 0;
    }
*/

    /* Display the floating point data */
    /*
       Serial.print("X: ");
       Serial.print(linaccel.x(), 4);
       Serial.print("\tY: ");
       Serial.print(linaccel.y(), 4);
       Serial.print("linaccel Z - avgAccelZeroZ: ");
       Serial.print(az, 4);
       Serial.print("\tgravZ: ");
       Serial.print(grav.z(), 4);
       Serial.println("");
     */

    /* Occassionally display the floating point data */
/*
    i+=1;
    if (i%50 == 0) {
        i = 0;
        Serial.print("Z in mm: ");
        Serial.println(z,4);
        Serial.print("\taz m/s^2: ");
        Serial.println(az,4);
        Serial.print("\tAvg Accel Z m/s^2: ");
        Serial.println(avgAccelZ,4);
        Serial.print("\tvz m/s: ");
        Serial.println(vz,4);
        Serial.print("\tdt s: ");
        Serial.println(dt,4);
    }
*/
    //Serial.print("\tY: ");
    //Serial.print(linaccel.y(),4);
    //Serial.print("\tZ: ");

    /* Optional: Display calibration status */
    //displayCalStatus();

    /* Optional: Display sensor status (debug only) */
    //displaySensorStatus();

    /* New line for the next sample */
    //Serial.println("");

    /* Wait the specified delay before requesting nex data */
    delay(BNO055_SAMPLERATE_DELAY_MS);
}
