#!/usr/bin/env python
import rospy

from std_msgs.msg import String

import sys, select, termios, tty


moveBindings = {
        'y':"yes",
        'n':"no",
           }



def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key



if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)
    
    pub = rospy.Publisher('keyboard_interrupt', String, queue_size=10)
    rospy.init_node('keyboard_interrupt_node')

    msg = ""

    try:
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            key = getKey()
            if key in moveBindings.keys():
                msg = moveBindings[key]

            else:
                msg = ""
                if (key == '\x03'):
                    break

            pub.publish(msg)

            rate.sleep()

    except:
        print "error"

    finally:
        msg = ""
        pub.publish(msg)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

