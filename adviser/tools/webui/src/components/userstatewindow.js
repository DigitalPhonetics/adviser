import React, {Component} from 'react';

// import './MessageItem.css';

class UserStateWindow extends Component {
    constructor(props) {
        super(props)
        this.state = {
            userstate: false,
            engagement: false
        }
    }

    componentDidMount() {
        this.props.socket.on("emotion", (user_state) => {
            console.log("User state from system: " + user_state);
            this.setState({ userstate: JSON.parse(user_state)}); })
        this.props.socket.on("engagement", (engagement) => {
                console.log("User engagement from system: " + engagement);
                this.setState({ engagement: engagement}); })
    }

    render() {
        return  <div className="UserStateWindow"> 
            {
                this.state.userstate != false && <div className='slotvaluecontainer'>
                            <div>Emotion:  {this.state.userstate['category']}</div><br/>
                            <div>
                                <div>Angry: {this.state.userstate['cateogry_probabilities']["Angry"]}</div>
                                <div>Happy: {this.state.userstate['cateogry_probabilities']["Happy"]}</div>
                                <div>Neutral: {this.state.userstate['cateogry_probabilities']["Neutral"]}</div>
                                <div>Sad: {this.state.userstate['cateogry_probabilities']["Sad"]}</div>
                            </div>
                        </div>
            } <br/>
            {
                this.state.engagement != false && <div className='slotvaluecontainer'>
                Engagement: <div className={this.state.engagement}>{this.state.engagement}</div>
            </div>
            }
        </div>
    }
}


export default UserStateWindow;