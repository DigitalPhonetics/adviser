#include "example_service.h"
#include <stdio.h>
#include <vector>
#include <chrono>

ExampleService::ExampleService(int sync_port, int sub_port, int pub_port, std::string ip_adress) : m_context(1) { // ZMQ Sockets need a context - initialized herre
    
    
    std::string sub_target = "tcp://" + ip_adress + ":" + std::to_string(sub_port);
    m_sub_socket = zmq::socket_t(m_context, ZMQ_SUB);
    m_sub_socket.setsockopt(ZMQ_SUBSCRIBE, TERMINATE_TOPIC, strlen(TERMINATE_TOPIC));
    m_sub_socket.setsockopt(ZMQ_SUBSCRIBE, "CONTENT", strlen("CONTENT"));
    m_sub_socket.bind(sub_target);

    std::string pub_target = "tcp://" + ip_adress + ":" + std::to_string(pub_port);
    m_pub_socket = zmq::socket_t(m_context, ZMQ_PUB);
    m_pub_socket.bind(pub_target);

    std::string sync_target = "tcp://" + ip_adress + ":" + std::to_string(sync_port);
    m_sync_socket = zmq::socket_t(m_context, ZMQ_REP);
    m_sync_socket.bind(sync_target);

    synchronize_with_dialogsystem();
    m_processing_thread = new std::thread(&ExampleService::receiver_loop, this);
}

void ExampleService::synchronize_with_dialogsystem()
{
    std::cout << "Registering with dialog system..." << std::endl;

    bool ready = false;
    while(!ready) {
        // blocking until received sync reply
        zmq::message_t reply;
        const auto ret = m_sync_socket.recv(reply);
        if(reply.to_string().compare("SYNC") == 0) {
            ready = true;
        }
    }
    m_sync_socket.send(zmq::str_buffer("ACK_SYNC"));

    std::cout << "Done" << std::endl;
}


void ExampleService::receiver_loop()
{
    std::cout << "Listening to messages from python..." << std::endl;
    bool listening = true;
    while(listening) {
        // blocking call: wait for message from dialog system
        std::pair<std::string, zmq::message_t> msg = recieve_message();
        std::string& topic = msg.first;
        if(topic.compare(TERMINATE_TOPIC) == 0) {
            // dialog system shutting down: stop listener loop
            listening = false;
            std::cout << "Stopped listening" << std::endl;
        } else {
            // normal message: call subscriber
            example_subscriber(topic, msg.second);
        }
    }
}

std::pair<std::string, zmq::message_t> ExampleService::recieve_message() {
    std::vector<zmq::message_t> recv_msgs;
    const auto ret = zmq::recv_multipart(m_sub_socket, std::back_inserter(recv_msgs));
    if (!ret) {
        std::cout << "Error while receiving" << std::endl;
        return std::make_pair("ERROR", zmq::message_t());
    }
    return std::make_pair(recv_msgs[0].to_string(), std::move(recv_msgs[1]));
}

void ExampleService::send_message(std::string topic, zmq::const_buffer &data)
{
    // send message
    std::array<zmq::const_buffer, 2> send_msgs = {
        zmq::buffer(topic),
        data
    };
    // if (!zmq::send_multipart(m_pub_socket, send_msgs))
    if (!zmq::send_multipart(m_pub_socket, send_msgs))
        std::cout << "Error while sending" << std::endl;
}

void ExampleService::example_subscriber(std::string& topic, zmq::message_t& msg)
{
    std::cout << "msg from python: " << msg.to_string() << std::endl;
    // reply 
    example_publisher();
}

void ExampleService::example_publisher() {
    auto content = zmq::str_buffer("HELLO FROM C++");
    send_message(std::string("CONTENT"), content);
}

int main() {
    ExampleService s = ExampleService();
    while(true) {
        // rest of your program
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
}