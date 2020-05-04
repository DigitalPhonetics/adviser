import React, {Component} from 'react';
import MessageList from './messagelist';
// import axios from 'axios';

// import '../App.css';

// axios.defaults.withCredentials = true;

class ChatWindow extends Component {

    constructor(props) {
        super(props);
        this.state = {
            input: "",
            messages: [
                // {
                //     message: "Welcome to ADvISER! How can I help you?", 
                //     party: "system"
                // }, {
                //     message: "test2",
                //     party: "user"
                // }
            ]
        }
    }

    sendEnterKey = (event) => {
        if(event.key === 'Enter') {
            this.send();
        }
    }

    textChange = (event) => {
        this.setState({input: event.target.value});
    }

    send_message = (message) => {
        this.props.socket.emit('user_utterance', message)
        // axios.post(`http://localhost:5000/chat`, {msg: message}).then(
        //     res => {
        //         console.log("Res.data");
        //         console.log(res.data);
        //         this.setState(state => {
        //             const new_msg_with_sys = state.messages.concat({
        //                 message: res.data['sys_utterance'],
        //                 party: "system"
        //             });
        //             return {
        //                 messages: new_msg_with_sys, 
        //                 input: ""
        //             }
        //         });
        //         this.props.receivedTurnInfo(res.data['turn_info']);
        //     }
        // )
    }

    componentDidMount() {
        this.props.socket.on("sys_utterance", (msg) => {
            console.log("Msg from system: " + msg);
            this.setState(state => {
                            const new_msg_with_sys = state.messages.concat({
                                message: msg,
                                party: "system"
                            });
                            return {
                                messages: new_msg_with_sys, 
                                input: ""
                            }
                        });});
        this.props.socket.on("user_utterance", (msg) => {
            console.log("Msg from system: " + msg);
            this.setState(state => {
                            const new_msg_with_usr = state.messages.concat({
                                message: msg,
                                party: "user"
                            });
                            return {
                                messages: new_msg_with_usr, 
                                input: ""
                            }
                        });});
        // axios.get(`http://localhost:5000/chat`, {}).then(
        //     res => {
        //         console.log(res.data['sys_utterance']);
        //         this.setState(state => {
        //             const new_msg_with_sys = state.messages.concat({
        //                 message: res.data['sys_utterance'],
        //                 party: "system"
        //             });
        //             return {
        //                 messages: new_msg_with_sys, 
        //                 input: ""
        //             }
        //         });
        //         this.props.receivedTurnInfo(res.data['turn_info'])     
        //     }
        // )
    }

    send = () => {
        const user_msg = this.state.input;
        console.log(user_msg);
        this.setState(state => {
            const new_msg_with_usr = state.messages.concat({
                message: state.input,
                party: "user"
            });
            return {
                messages: new_msg_with_usr, 
                input: ""
            }
        });
        this.send_message(this.state.input);
    }

    render() {
        return <div className="chatwindow">
            <div className="messagelist">
                <MessageList messages={this.state.messages} />
            </div>
            <div className="chatbox">
                <input className="input" autoFocus onKeyDown={this.sendEnterKey} onChange={this.textChange} value={this.state.input}/>
                <button className="button" onClick={this.send}>send</button>
            </div>
        </div>
    }
}

export default ChatWindow;