#include <iostream>
#include <string>
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <thread>

class ExampleService {
    public:
        ExampleService(int sync_port = 6006, int sub_port = 6007, int pub_port = 6008, std::string ip_adress = "127.0.0.1");

        // message handling
        void example_subscriber(std::string& topic, zmq::message_t& msg);
        void example_publisher();

    private:
        zmq::context_t m_context;

        zmq::socket_t m_sync_socket;
        zmq::socket_t m_pub_socket;
        zmq::socket_t m_sub_socket;

        const char* TERMINATE_TOPIC = "CPPService/TERMINATE";

        void synchronize_with_dialogsystem();

        // messaging functionallity
        std::pair<std::string, zmq::message_t> recieve_message();
        void send_message(std::string topic, zmq::const_buffer &data);

        // service state
        void receiver_loop();
        std::thread* m_processing_thread;

};