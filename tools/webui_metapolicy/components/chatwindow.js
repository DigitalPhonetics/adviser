import React, {Component} from 'react';
import MessageList from './messagelist';
import axios from 'axios';

axios.defaults.withCredentials = true;

class ChatWindow extends Component {

    constructor(props) {
        super(props);
        this.state = {
            input: "",
            messages: [
                // {
                //     message: "Welcome to ADvISER! How can I help you?", 
                //     party: "system"
                // },// {
                //     message: "test2",
                //     party: "user"
                // }
            ]
        }
    }

    scrollToBottom = () => {
        this.messagesEnd.scrollIntoView({ behavior: "smooth" });
    }
    
    sendEnterKey = (event) => {
        if(event.key == 'Enter') {
            this.send();
        }
    }

    textChange = (event) => {
        this.setState({input: event.target.value});
    }

    send_message = (message) => {
        axios.post(`http://localhost:5000/chat`, {msg: message}).then(
            res => {
                console.log("Res.data");
                console.log(res.data);
                this.setState(state => {
                    const new_msg_with_sys = state.messages.concat({
                        message: res.data['sys_utterance'],
                        party: "system"
                    });
                    return {
                        messages: new_msg_with_sys, 
                        input: ""
                    }
                });
                this.props.receivedTurnInfo(res.data['turn_info']);
            }
        )
    }

    componentDidMount() {
        axios.get(`http://localhost:5000/chat`, {}).then(
            res => {
                console.log(res.data['sys_utterance']);
                this.setState(state => {
                    const new_msg_with_sys = state.messages.concat({
                        message: res.data['sys_utterance'],
                        party: "system"
                    });
                    return {
                        messages: new_msg_with_sys, 
                        input: ""
                    }
                });
                this.props.receivedTurnInfo(res.data['turn_info'])     
            }
        )
        this.scrollToBottom();
    }

    componentDidUpdate() {
        this.scrollToBottom();
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
                <div style={{ float:"left", clear: "both" }}
                ref={(el) => { this.messagesEnd = el; }}>
                </div>
            </div>  
            <div className="chatbox">
                <input className="input" autoFocus onKeyDown={this.sendEnterKey} onChange={this.textChange} value={this.state.input}/>
                <button className="button" onClick={this.send}>send</button>
            </div>
        <style jsx>{`
            .chatwindow {
                display: grid;
                height: 100%;
                grid-template-rows: 1fr auto;
                max-height: 100%
            }
            .messagelist {
                overflow-y: scroll; 
                overflow-x: hidden;
                max-height: calc(100vh - 35px);
                grid-row: 1;
            }
            .chatbox {
                align-self: end;
                display: flex;
                width: 100%;
                height: 25px;
                grid-row: 2;
                background: white;
                z-index: 10; 
            }
            .input {
                border-radius: 5px;
                border: 0px;
                padding: 7px;
                align-self: stretch;
                flex: 1;
            }
            .button {
                align-self: end;
                background: #00BEFF;
                border: 0px;
                border-radius: 7px;
                padding: 5px;
                font-size: 14px;
                flex: 0;
            }
            `}
        </style>
        </div>
    }
}

export default ChatWindow;